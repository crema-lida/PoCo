from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional
import hashlib
import multiprocessing as mp
import os

import numpy as np
from numpy.typing import NDArray
import torch

from unicore.data import Dictionary

from .coords import prepare_mm_polymer_conformers
from .models.MMPolymer import MMPolymerModel
from .models.PolymerSmilesTokenization import PolymerSmilesTokenizer


def _mm_polymer_symbol_to_idx(dictionary, symbol: str) -> int:
    if hasattr(dictionary, "index"):
        return dictionary.index(symbol)
    indices = getattr(dictionary, "indices", None)
    if indices is not None:
        return indices.get(symbol, dictionary.unk())
    return dictionary.unk()


def _mm_polymer_filter_atoms(
    atoms,
    coords_list,
    remove_hydrogen: bool,
    remove_polar_hydrogen: bool,
):
    atoms = np.array(atoms)
    if remove_hydrogen:
        mask = atoms != "H"
        atoms = atoms[mask]
        coords_list = [coords[mask] for coords in coords_list]
    elif remove_polar_hydrogen:
        end_idx = 0
        for atom in atoms[::-1]:
            if atom != "H":
                break
            end_idx += 1
        if end_idx:
            atoms = atoms[:-end_idx]
            coords_list = [coords[:-end_idx] for coords in coords_list]
    return atoms, coords_list


def _mm_polymer_crop_and_normalize(atoms, coords_list, max_atoms: int, seed: int):
    if max_atoms and len(atoms) > max_atoms:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[indices]
        coords_list = [coords[indices] for coords in coords_list]

    if len(atoms) > 0:
        coords_list = [coords - coords.mean(axis=0) for coords in coords_list]
    return atoms, coords_list


def _mm_polymer_build_sample(atoms, coords, dictionary):
    token_ids = [_mm_polymer_symbol_to_idx(dictionary, atom) for atom in atoms]
    token_ids = [dictionary.bos()] + token_ids + [dictionary.eos()]
    tokens = torch.tensor(token_ids, dtype=torch.long)

    coords = np.vstack(
        [np.zeros((1, 3), dtype=np.float32), coords, np.zeros((1, 3), dtype=np.float32)]
    )
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1).astype(np.float32)
    dist_tensor = torch.from_numpy(dist)

    num_types = len(dictionary)
    edge_type = tokens.view(-1, 1) * num_types + tokens.view(1, -1)

    return tokens, dist_tensor, edge_type


def _mm_polymer_pad_1d(values, pad_value: int) -> torch.Tensor:
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_value)
    for i, v in enumerate(values):
        res[i, : v.size(0)] = v
    return res


def _mm_polymer_pad_2d(values, pad_value: int) -> torch.Tensor:
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size, size).fill_(pad_value)
    for i, v in enumerate(values):
        res[i, : v.size(0), : v.size(1)] = v
    return res


def _prepare_mm_polymer_smiles(args):
    smiles, conf_size, max_atoms, remove_hydrogen, remove_polar_hydrogen = args
    try:
        atoms, coords_list = prepare_mm_polymer_conformers(smiles, conf_size=conf_size)
        atoms, coords_list = _mm_polymer_filter_atoms(
            atoms,
            coords_list,
            remove_hydrogen=remove_hydrogen,
            remove_polar_hydrogen=remove_polar_hydrogen,
        )
        seed = int.from_bytes(hashlib.md5(smiles.encode("utf-8")).digest()[:4], "little")
        atoms, coords_list = _mm_polymer_crop_and_normalize(
            atoms,
            coords_list,
            max_atoms=max_atoms,
            seed=seed,
        )
        if len(atoms) == 0:
            return smiles, None, None, "empty atoms"
        return smiles, atoms, coords_list, None
    except Exception as exc:
        return smiles, None, None, str(exc)


def build_mm_polymer_encoder(model_path: Path) -> Callable[[list[str]], NDArray]:
    dictionary = Dictionary.load(str(model_path / "dict.txt"))
    dictionary.add_symbol("[MASK]", is_special=True)

    args = SimpleNamespace()
    model = MMPolymerModel(args, dictionary)
    state = torch.load(model_path / "pretrain.pt")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval().cuda()

    tokenizer = model.tokenizer
    output_dim = model.config.hidden_size + args.encoder_embed_dim

    @torch.no_grad
    def encode(
        sentences: list[str],
        pooling: str = "mean",
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        conf_size: int = 1,
        max_atoms: int = 256,
        remove_hydrogen: bool = False,
        remove_polar_hydrogen: bool = True,
        **kwargs,
    ) -> NDArray:
        if not sentences:
            return np.empty((0, output_dim), dtype=np.float32)

        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 1)

        if num_workers > 1 and len(sentences) > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                results = pool.map(
                    _prepare_mm_polymer_smiles,
                    [
                        (smiles, conf_size, max_atoms, remove_hydrogen, remove_polar_hydrogen)
                        for smiles in sentences
                    ],
                )
        else:
            results = [
                _prepare_mm_polymer_smiles(
                    (smiles, conf_size, max_atoms, remove_hydrogen, remove_polar_hydrogen)
                )
                for smiles in sentences
            ]

        items = []
        group_sizes = []
        pad_idx = dictionary.pad()

        smiles_list = []
        prepared = []
        for smiles, atoms, coords_list, err in results:
            if err:
                raise ValueError(f"MMPolymer 3D failed to build atoms for: {smiles} ({err})")
            smiles_list.append(str(smiles))
            prepared.append((atoms, coords_list))

        encoding = tokenizer(
            smiles_list,
            add_special_tokens=True,
            max_length=411,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        for idx, (atoms, coords_list) in enumerate(prepared):
            input_ids = encoding["input_ids"][idx]
            attention_mask = encoding["attention_mask"][idx]

            for coords in coords_list:
                tokens, dist_tensor, edge_type = _mm_polymer_build_sample(
                    atoms,
                    coords,
                    dictionary,
                )
                items.append(
                    (
                        tokens,
                        dist_tensor,
                        edge_type,
                        input_ids,
                        attention_mask,
                    )
                )
            group_sizes.append(len(coords_list))

        outputs = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            src_tokens = _mm_polymer_pad_1d([b[0] for b in batch], pad_idx)
            src_distance = _mm_polymer_pad_2d([b[1] for b in batch], 0)
            src_edge_type = _mm_polymer_pad_2d([b[2] for b in batch], 0)
            src_input_ids = torch.stack([b[3] for b in batch])
            src_attention_mask = torch.stack([b[4] for b in batch])

            seq_output, space_output = model(
                src_tokens=src_tokens.to("cuda"),
                src_distance=src_distance.to("cuda"),
                src_edge_type=src_edge_type.to("cuda"),
                src_input_ids=src_input_ids.to("cuda"),
                src_attention_mask=src_attention_mask.to("cuda"),
                pooling=pooling,
            )
            outputs.append(torch.cat([seq_output, space_output], dim=-1))

        embeds = torch.cat(outputs, dim=0)
        if any(size > 1 for size in group_sizes):
            averaged = []
            offset = 0
            for size in group_sizes:
                averaged.append(embeds[offset : offset + size].mean(dim=0))
                offset += size
            embeds = torch.stack(averaged, dim=0)
        return embeds.cpu().numpy()

    return encode
