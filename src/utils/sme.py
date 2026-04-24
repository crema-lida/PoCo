import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
import multiprocessing as mp
from typing import Optional

import pandas as pd
from scipy.stats import t as student_t
import torch
from tqdm import tqdm

from .gram_masking import make_gram_fragments
from .brics_masking import make_brics_fragments

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_OCCURRENCE_COLUMNS = [
    "sample_idx",
    "smiles",
    "fragment",
    "y_base",
    "y_sub",
    "delta",
    "atom_group",
]

_STATS_COLUMNS = [
    "fragment",
    "n",
    "mean_delta",
    "std_delta",
    "t_stat",
    "pval",
    "pval_adj",
    "neglog10_p",
    "neglog10_fdr",
]


def collate_pad(examples: list[dict[str, torch.Tensor]], pad_id: int):
    max_len = max(len(x["input_ids"]) for x in examples)
    B = len(examples)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, ex in enumerate(examples):
        L = len(ex["input_ids"])
        input_ids[i, :L] = ex["input_ids"]
        attn_mask[i, :L] = ex["attention_mask"]
    return {"input_ids": input_ids, "attention_mask": attn_mask}


@torch.no_grad()
def predict(model, examples, pad_id, batch_size, device):
    logits_all = []
    for i in tqdm(range(0, len(examples), batch_size), desc="Predicting", leave=False):
        batch = collate_pad(examples[i:i + batch_size], pad_id)
        inp = {k: v.to(device) for k, v in batch.items()}
        logits = model(**inp)
        logits = logits.squeeze(-1)
        logits_all.append(logits.cpu().numpy())
    if not logits_all:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(logits_all, axis=0)


@torch.no_grad()
def encode(model, inputs, pad_id, batch_size, device) -> list[torch.Tensor]:
    embeds_all = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="Encoding", leave=False):
        batch = collate_pad(inputs[i:i + batch_size], pad_id)
        inp = {k: v.to(device) for k, v in batch.items()}
        embeds = model(**inp).last_hidden_state.cpu()
        for emb, mask in zip(embeds, batch["attention_mask"]):
            L = mask.sum().item()
            embeds_all.append(emb[:L])
    return embeds_all


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    m = pvals.size
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    ranks = np.arange(1, m + 1)
    adjusted = sorted_p * m / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def compute_fragment_stats(df_occurrence: pd.DataFrame) -> pd.DataFrame:
    occ_subset = df_occurrence.loc[:, ["fragment", "delta"]].dropna(subset=["fragment", "delta"])

    agg = (
        occ_subset.groupby("fragment")["delta"]
        .agg(n="count", mean_delta="mean", std_delta="std")
        .reset_index()
    )
    if agg.empty:
        return pd.DataFrame(columns=_STATS_COLUMNS)

    df_stats = agg.copy()
    df_stats["n"] = df_stats["n"].astype(int)
    df_stats["mean_delta"] = df_stats["mean_delta"].astype(float)
    df_stats["std_delta"] = df_stats["std_delta"].astype(float)

    df_stats["t_stat"] = np.nan
    df_stats["pval"] = np.nan

    eligible = (df_stats["n"] > 1) & df_stats["std_delta"].notna()
    if eligible.any():
        subset = df_stats.loc[eligible, ["mean_delta", "std_delta", "n"]]

        nonzero_std = subset["std_delta"] > 0
        if nonzero_std.any():
            nonzero = subset.loc[nonzero_std]
            stderr = nonzero["std_delta"] / np.sqrt(nonzero["n"])
            t_stats = nonzero["mean_delta"] / stderr
            df_stats.loc[nonzero.index, "t_stat"] = t_stats.to_numpy()
            df_stats.loc[nonzero.index, "pval"] = 2.0 * student_t.sf(
                np.abs(t_stats.to_numpy()),
                df=nonzero["n"].to_numpy() - 1,
            )

        zero_std = ~nonzero_std
        if zero_std.any():
            zero_var = subset.loc[zero_std]
            means = zero_var["mean_delta"].to_numpy()
            zero_mean = np.isclose(means, 0.0)
            df_stats.loc[zero_var.index, "t_stat"] = np.where(
                zero_mean,
                0.0,
                np.copysign(np.inf, means),
            )
            df_stats.loc[zero_var.index, "pval"] = np.where(zero_mean, 1.0, 0.0)

    valid = df_stats["pval"].notna()
    df_stats["pval_adj"] = np.nan
    if valid.any():
        adj = bh_fdr(df_stats.loc[valid, "pval"].to_numpy())
        df_stats.loc[valid, "pval_adj"] = adj

    pval = df_stats["pval"]
    pval_adj = df_stats["pval_adj"]
    df_stats["neglog10_p"] = np.where(
        pval.notna(),
        -np.log10(pval.clip(lower=1e-300)),
        np.nan,
    )
    df_stats["neglog10_fdr"] = np.where(
        pval_adj.notna(),
        -np.log10(pval_adj.clip(lower=1e-300)),
        np.nan,
    )

    return df_stats.loc[:, _STATS_COLUMNS]


def _build_gram_fragments(args):
    smiles, tokens, embeds = args
    result = make_gram_fragments(smiles, tokens, embeds)
    frag_ids = result.get("frag_ids", [])
    fragments = result.get("fragments", [])
    atom_groups = result.get("atom_groups", [])
    return frag_ids, fragments, atom_groups


def _build_brics_fragments(args):
    smiles, tokens, _embeds = args
    result = make_brics_fragments(smiles, tokens)
    frag_ids = result.get("frag_ids", [])
    fragments = result.get("fragments", [])
    atom_groups = result.get("atom_groups", [])
    return frag_ids, fragments, atom_groups


_FRAGMENT_BUILDERS = {
    "gram": _build_gram_fragments,
    "brics": _build_brics_fragments,
}


def _tokenize_chunk(tokenizer, smiles_chunk):
    enc = tokenizer(smiles_chunk, return_token_type_ids=False)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    base_examples = [
        {
            "input_ids": torch.as_tensor(ids),
            "attention_mask": torch.as_tensor(mask),
        }
        for ids, mask in zip(input_ids, attention_mask)
    ]
    tokens_all = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
    return base_examples, tokens_all


@contextmanager
def _thread_env_guard(num_workers: int):
    if num_workers <= 1:
        yield
        return

    backup = {name: os.environ.get(name) for name in _THREAD_ENV_VARS}
    for name in _THREAD_ENV_VARS:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in backup.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _aggregate_occurrences(
    masked_meta,
    sub_preds,
    intra_mol_mean: bool = True,
) -> list[dict[str, object]]:
    aggregated: dict[tuple[object, ...], dict[str, object]] = {}
    for occ_idx, ((idx, smiles, frag_label, atom_group, base_pred), sub_pred) in enumerate(
        zip(masked_meta, sub_preds)
    ):
        sub_pred = float(sub_pred)
        delta = base_pred - sub_pred
        key = (idx, frag_label) if intra_mol_mean else (idx, frag_label, occ_idx)
        entry = aggregated.get(key)
        if entry is None:
            entry = aggregated[key] = {
                "sample_idx": idx,
                "smiles": smiles,
                "fragment": frag_label,
                "y_base": base_pred,
                "sum_y_sub": 0.0,
                "sum_delta": 0.0,
                "count": 0,
                "atom_group": atom_group,
            }
        entry["sum_y_sub"] = entry["sum_y_sub"] + sub_pred
        entry["sum_delta"] = entry["sum_delta"] + delta
        entry["count"] = entry["count"] + 1

    return [
        {
            "sample_idx": entry["sample_idx"],
            "smiles": entry["smiles"],
            "fragment": entry["fragment"],
            "y_base": entry["y_base"],
            "y_sub": entry["sum_y_sub"] / entry["count"],
            "delta": entry["sum_delta"] / entry["count"],
            "atom_group": str(entry["atom_group"]),
        }
        for entry in aggregated.values()
    ]


def run_sme(
    smiles_list: list[str],
    tokenizer,
    model,
    method: str = "gram",
    batch_size: int = 64,
    device: Optional[str] = None,
    chunk_size: int = 512,
    num_workers: Optional[int] = 1,
    intra_mol_mean: bool = True,
    return_stats: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run substructure masking and compute fragment statistics fully in memory with pandas.

    Args:
        smiles_list: List of SMILES strings.
        tokenizer: Hugging Face tokenizer.
        model: Predictive model returning scalar outputs.
        batch_size: Forward-pass batch size.
        device: Torch device (auto-detects when None).
        chunk_size: Number of molecules processed per iteration.
        num_workers: Worker processes for fragment/mask building per chunk. Use <= 1 to disable.
            When None, uses max(1, cpu_count - 1).
        intra_mol_mean: Whether to average repeated occurrences of the same fragment within
            one molecule.
        return_stats: Whether to compute and return df_stats.

    Returns:
        (df_occurrence, df_stats or None)
    """
    method = method.lower()
    frag_builder = _FRAGMENT_BUILDERS.get(method)
    if frag_builder is None:
        raise ValueError(f"Unsupported fragmentation method: {method}")
    use_embeds = method == "gram"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    encoder = model.encoder

    pad_id = tokenizer.pad_token_id

    occurrence_rows: list[dict[str, object]] = []

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) - 1)

    with _thread_env_guard(num_workers):
        executor_cm = (
            ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp.get_context("spawn"),
            )
            if num_workers > 1
            else nullcontext()
        )
        with executor_cm as executor:
            for chunk_start in tqdm(range(0, len(smiles_list), chunk_size), desc="SME chunks"):
                smiles_chunk = smiles_list[chunk_start:chunk_start + chunk_size]

                base_examples, tokens_all = _tokenize_chunk(tokenizer, smiles_chunk)
                embeds_all = (
                    encode(encoder, base_examples, pad_id, batch_size, device)
                    if use_embeds
                    else [None] * len(smiles_chunk)
                )
                base_preds = predict(model, base_examples, pad_id, batch_size, device)

                build_inputs = list(zip(smiles_chunk, tokens_all, embeds_all))
                if executor is None:
                    frag_results = map(frag_builder, build_inputs)
                else:
                    chunksize = max(1, len(build_inputs) // (num_workers * 4)) if build_inputs else 1
                    frag_results = executor.map(frag_builder, build_inputs, chunksize=chunksize)

                masked_examples = []
                masked_meta = []
                for local_idx, (frag_ids, fragments, atom_groups) in enumerate(frag_results):
                    if not frag_ids:
                        continue
                    base_inputs = base_examples[local_idx]
                    base_pred = float(base_preds[local_idx])
                    global_idx = chunk_start + local_idx
                    for frag_pos, frag_label, atom_group in zip(frag_ids, fragments, atom_groups):
                        mask = base_inputs["attention_mask"].clone()
                        mask[frag_pos] = 0
                        masked_examples.append(
                            {
                                "input_ids": base_inputs["input_ids"],
                                "attention_mask": mask,
                            }
                        )
                        masked_meta.append(
                            (
                                global_idx,
                                smiles_chunk[local_idx],
                                frag_label,
                                atom_group,
                                base_pred,
                            )
                        )

                if not masked_examples:
                    continue

                sub_preds = predict(model, masked_examples, pad_id, batch_size, device)
                occurrence_rows.extend(
                    _aggregate_occurrences(
                        masked_meta,
                        sub_preds,
                        intra_mol_mean=intra_mol_mean,
                    )
                )

    df_occurrence = pd.DataFrame(occurrence_rows, columns=_OCCURRENCE_COLUMNS)
    df_stats = compute_fragment_stats(df_occurrence) if return_stats else None

    return df_occurrence, df_stats
