from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional
import multiprocessing as mp
import os
import pickle

import numpy as np
from numpy.typing import NDArray
import torch

from .data.vocab import Vocab
from .models.light import LiGhTPredictor as LiGhT
from .data.smiles2g_light import PolyGraphBuilderFinetune
from .utils.function import load_config, preprocess_batch_light
from .utils.aug import generate_multimer_smiles
from .utils.features import _fp_md_from_smiles
import dgl


def _periogt_fp_md_worker(smiles: str):
    try:
        multimer = generate_multimer_smiles(3, smiles, replace_dummy_atoms=True)
    except Exception:
        try:
            multimer = generate_multimer_smiles(3, smiles, replace_dummy_atoms=False)
        except Exception as exc:
            return smiles, None, None, f"multimer: {exc}"

    try:
        _, fp, md = _fp_md_from_smiles(multimer)
    except Exception as exc:
        return smiles, None, None, f"features: {exc}"

    if fp is None or md is None:
        return smiles, None, None, "features: None"

    return smiles, fp, md, None


def build_periogt_encoder(model_path: Path) -> Callable[[list[str]], NDArray]:
    config_args = SimpleNamespace(backbone="light", config="base")
    config = load_config(config_args, file_path=str(model_path / "config.yaml"))
    vocab = Vocab()
    model = LiGhT(
        d_node_feats=config["d_node_feats"],
        d_edge_feats=config["d_edge_feats"],
        d_g_feats=config["d_g_feats"],
        d_cl_feats=config["d_cl_feats"],
        d_fp_feats=1191,
        d_md_feats=1613,
        d_hpath_ratio=config["d_hpath_ratio"],
        n_mol_layers=config["n_mol_layers"],
        path_length=config["path_length"],
        n_heads=config["n_heads"],
        n_ffn_dense_layers=config["n_ffn_dense_layers"],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size,
    )
    ckpt_path = model_path / "checkpoints/pretrained/light/base.pth"
    state = torch.load(ckpt_path)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval().cuda()

    scaler_path = model_path / "datasets/pretrain/scaler_all.pkl"
    with open(scaler_path, "rb") as handle:
        scaler = pickle.load(handle)

    builder_args = SimpleNamespace(use_prompt=False, device="cuda", max_prompt=0)
    builder = PolyGraphBuilderFinetune(builder_args, scaler)
    output_dim = config["d_g_feats"] * 3

    @torch.no_grad
    def encode(
        sentences: list[str],
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs,
    ) -> NDArray:
        if not sentences:
            return np.empty((0, output_dim), dtype=np.float32)

        graphs = []
        for smiles in sentences:
            graph = builder.build(smiles)
            if graph is None:
                raise ValueError(f"Invalid SMILES for PerioGT: {smiles}")
            graphs.append(graph)

        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 1)

        fp_list = []
        md_list = []
        if num_workers > 1 and len(sentences) > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                results = pool.map(_periogt_fp_md_worker, sentences)
            for smi, fp, md, err in results:
                if err:
                    raise ValueError(f"PerioGT failed to compute features for: {smi} ({err})")
                fp_list.append(fp)
                md_list.append(md)
        else:
            for smiles in sentences:
                smi, fp, md, err = _periogt_fp_md_worker(smiles)
                if err:
                    raise ValueError(f"PerioGT failed to compute features for: {smi} ({err})")
                fp_list.append(fp)
                md_list.append(md)

        fps = np.array(fp_list, dtype=np.float32)
        mds = np.array(md_list, dtype=np.float32)
        mds = scaler.transform(mds).astype(np.float32)

        outputs = []
        for i in range(0, len(sentences), batch_size):
            batch_graph = dgl.batch(graphs[i : i + batch_size])
            batch_graph.edata["path"][:, :] = preprocess_batch_light(
                batch_graph.batch_num_nodes(),
                batch_graph.batch_num_edges(),
                batch_graph.edata["path"][:, :],
            )
            batch_graph = batch_graph.to("cuda")
            fp = torch.from_numpy(fps[i : i + batch_size]).to("cuda")
            md = torch.from_numpy(mds[i : i + batch_size]).to("cuda")
            embeds = model.generate_features(batch_graph, fp, md)
            outputs.append(embeds)
        return torch.cat(outputs).cpu().numpy()

    return encode
