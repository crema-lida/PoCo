from __future__ import annotations

from pathlib import Path
from typing import Callable
import sys

import numpy as np
from numpy.typing import NDArray
import torch

from .distance import mol_dis_sim
from .feature import mol_feature
from .rbf_direction import calculate_edge_fea


CUTOFFS = ["0-2", "0-2.5", "0-3", "0-3.5", "0-4"]
NUM_NET = 5
EMBED_DIM = 64


def _get_element_index(mat_0_1, mat_dis):
    arr_index = []
    arr_dis = []
    for i in range(len(mat_0_1)):
        for j in range(len(mat_0_1[i])):
            if float(mat_0_1[i][j]) == 1.0:
                arr_index.append([i, j])
                arr_dis.append(mat_dis[i][j])
    return arr_index, arr_dis


def _get_edge_index(edge_index, edge_dis, arr_coor):
    edge_num = len(edge_index)
    ed_arr_index_num = []
    arr_edge_representation = []

    for i in range(edge_num):
        rbf_rep = calculate_edge_fea.rbf_vector(edge_dis[i])
        direction_rep = calculate_edge_fea.direction_vector(arr_coor, edge_index[i])
        edge_representation = rbf_rep + direction_rep

        for j in range(edge_num):
            if i == j:
                ed_arr_index_num.append([i, j])
            elif len(list(set(edge_index[i]) & set(edge_index[j]))) == 1:
                ed_arr_index_num.append([i, j])

        arr_edge_representation.append(edge_representation)

    return ed_arr_index_num, arr_edge_representation


def _smiles_to_coor(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem import rdDepictor
    except ImportError as exc:
        raise ImportError("Mol-TDL encoder requires RDKit (rdkit).") from exc

    normalized = (smiles
                  .replace('[*]=', 'C=')
                  .replace('=[*]', '=C')
                  .replace('[*]#', 'C#')
                  .replace('#[*]', '#C')
                  .replace("[*]", "C"))
    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        raise ValueError(f"Invalid SMILES for Mol-TDL: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    params.useRandomCoords = True
    params.maxIterations = 200
    if AllChem.EmbedMolecule(mol, params) != 0:
        rdDepictor.Compute2DCoords(mol)
        print(f"Warning: 3D embedding failed for SMILES: {smiles}, using 2D coordinates instead.")

    conf = mol.GetConformer()
    arr_coor = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        arr_coor.append(f"{atom.GetSymbol()}\t{pos.x}\t{pos.y}\t{pos.z}")
    return arr_coor


def _smile_to_graph3(arr_coor, list_cutoff):
    c_size = []
    features = []
    edge_indexs = []

    edge_edge_feats = []
    edge_edge_nums = []

    for cutoff in list_cutoff:
        arr_cutoff = cutoff.split("-")
        drug_dis, drug_dis_real, drug_atoms = mol_dis_sim.Calculate_distance(arr_coor, arr_cutoff)
        drug_feat = mol_feature.Calculate_feature(drug_dis_real, drug_atoms)
        edge_index, edge_dis = _get_element_index(drug_dis, drug_dis_real)
        edge_edge_num, edge_edge_feat = _get_edge_index(edge_index, edge_dis, arr_coor)

        c_size.append(len(arr_coor))
        features.append(drug_feat)
        edge_indexs.append(edge_index)

        edge_edge_feats.append(edge_edge_feat)
        edge_edge_nums.append(edge_edge_num)

    return c_size, features, edge_indexs, edge_edge_feats, edge_edge_nums


def _ensure_nonempty(x: torch.Tensor, feat_dim: int) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros((1, feat_dim), dtype=torch.float)
    return x


def _build_graphs(smiles: str):
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("Mol-TDL encoder requires torch_geometric.") from exc

    arr_coor = _smiles_to_coor(smiles)
    c_size, features, edge_indexs, edge_edge_feats, edge_edge_nums = _smile_to_graph3(
        arr_coor, CUTOFFS
    )

    data_list = []
    data2_list = []
    for idx in range(NUM_NET):
        x = torch.tensor(features[idx], dtype=torch.float)
        x = _ensure_nonempty(x, 62)
        if edge_indexs[idx]:
            edge_index = torch.tensor(edge_indexs[idx], dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))

        edge_x = torch.tensor(edge_edge_feats[idx], dtype=torch.float)
        edge_x = _ensure_nonempty(edge_x, 10)
        if edge_edge_nums[idx]:
            edge_edge_index = torch.tensor(edge_edge_nums[idx], dtype=torch.long).t().contiguous()
        else:
            edge_edge_index = torch.empty((2, 0), dtype=torch.long)
        data2_list.append(Data(x=edge_x, edge_index=edge_edge_index))

    return data_list, data2_list


def _is_state_dict(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if not obj:
        return False
    return all(torch.is_tensor(v) for v in obj.values())


def _extract_state_dict(obj):
    if _is_state_dict(obj):
        return obj
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "net", "encoder"):
            if key in obj and _is_state_dict(obj[key]):
                return obj[key]
    raise ValueError("Unable to find a state dict in checkpoint")


def _load_model(model_path: Path):
    sys_path = str(model_path)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    from importlib import import_module, util

    if util.find_spec("models.gcn") is None:
        raise ImportError("Unable to import Mol-TDL models.gcn for checkpoint loading.")
    GCNNet = import_module("models.gcn").GCNNet

    if util.find_spec("torch_geometric.nn.conv.utils.inspector") is None:
        import types

        module_name = "torch_geometric.nn.conv.utils.inspector"
        parent_name = "torch_geometric.nn.conv.utils"
        if parent_name not in sys.modules:
            sys.modules[parent_name] = types.ModuleType(parent_name)
        stub = types.ModuleType(module_name)

        class Inspector:  # noqa: D401
            """Stub for older torch_geometric Inspector used in pickled checkpoints."""

            def implements(self, *_args, **_kwargs):
                return False

        stub.Inspector = Inspector
        sys.modules[module_name] = stub
    else:
        import_module("torch_geometric.nn.conv.utils.inspector")

    if not hasattr(torch.nn.Module, "_lazy_load_hook"):
        torch.nn.Module._lazy_load_hook = None
    if not hasattr(torch.nn.Linear, "_lazy_load_hook"):
        torch.nn.Linear._lazy_load_hook = None

    checkpoint_path = model_path / "model_pretrain_100k.pt"
    from torch.serialization import add_safe_globals

    safe_types = [
        GCNNet,
        torch.nn.ModuleList,
        torch.nn.Linear,
        torch.nn.Dropout,
        torch.nn.ReLU,
    ]
    if util.find_spec("torch_geometric.nn") is not None:
        GCNConv = import_module("torch_geometric.nn").GCNConv
        safe_types.append(GCNConv)
    add_safe_globals(safe_types)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(state, torch.nn.Module):
        state_dict = state.state_dict()
    elif isinstance(state, dict):
        state_dict = _extract_state_dict(state)
    else:
        raise ValueError("Unexpected Mol-TDL checkpoint format")

    model = GCNNet()
    model.load_state_dict(state_dict, strict=False)
    return model


def build_moltdl_encoder(model_path: Path) -> Callable[[list[str]], NDArray]:
    model = _load_model(model_path)
    model.eval().cuda()

    try:
        from torch_geometric.data import Batch
    except ImportError as exc:
        raise ImportError("Mol-TDL encoder requires torch_geometric.") from exc

    @torch.no_grad
    def encode(
        sentences: list[str],
        batch_size: int = 16,
        **kwargs,
    ) -> NDArray:
        if not sentences:
            return np.empty((0, EMBED_DIM), dtype=np.float32)

        outputs = []
        device = torch.device("cuda")
        for i in range(0, len(sentences), batch_size):
            batch_smiles = sentences[i : i + batch_size]
            batch_lists = [[] for _ in range(NUM_NET)]
            batch_lists2 = [[] for _ in range(NUM_NET)]
            for smiles in batch_smiles:
                data_list, data2_list = _build_graphs(smiles)
                for idx in range(NUM_NET):
                    batch_lists[idx].append(data_list[idx])
                    batch_lists2[idx].append(data2_list[idx])

            batched = [Batch.from_data_list(batch_lists[idx]) for idx in range(NUM_NET)]
            batched2 = [Batch.from_data_list(batch_lists2[idx]) for idx in range(NUM_NET)]
            embeds = model(batched, batched2, device)
            outputs.append(embeds)

        return torch.cat(outputs, dim=0).cpu().numpy()

    return encode
