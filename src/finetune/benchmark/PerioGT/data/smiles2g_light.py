from functools import partial
from itertools import permutations

import dgl
import numpy as np
import torch
from rdkit import Chem
from dgllife.utils.featurizers import (
    ConcatFeaturizer,
    bond_type_one_hot,
    bond_is_conjugated,
    bond_is_in_ring,
    bond_stereo_one_hot,
    atomic_number_one_hot,
    atom_degree_one_hot,
    atom_formal_charge,
    atom_num_radical_electrons_one_hot,
    atom_hybridization_one_hot,
    atom_is_aromatic,
    atom_total_num_H_one_hot,
    atom_is_chiral_center,
    atom_chirality_type_one_hot,
    atom_mass,
)
import networkx as nx

from ..data.constants import (
    VIRTUAL_ATOM_INDICATOR,
    VIRTUAL_ATOM_FEATURE_PLACEHOLDER,
    VIRTUAL_BOND_FEATURE_PLACEHOLDER,
    VIRTUAL_PATH_INDICATOR,
)


class PolyGraphBuilderFinetune:
    def __init__(self, args, scaler, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.atom_featurizer = ConcatFeaturizer([  # 138
            partial(atomic_number_one_hot, allowable_set=list(range(0, 101)), encode_unknown=True),  # 102
            partial(atom_degree_one_hot, encode_unknown=True),  # 12
            atom_formal_charge,  # 1
            partial(atom_num_radical_electrons_one_hot, encode_unknown=True),  # 6
            partial(atom_hybridization_one_hot, encode_unknown=True),  # 6
            atom_is_aromatic,  # 1
            partial(atom_total_num_H_one_hot, encode_unknown=True),  # 6
            atom_is_chiral_center,  # 1
            atom_chirality_type_one_hot,  # 2
            atom_mass,  # 1
        ])
        self.bond_featurizer = ConcatFeaturizer([  # 14
            partial(bond_type_one_hot, encode_unknown=True),  # 5
            bond_is_conjugated,  # 1
            bond_is_in_ring,  # 1
            partial(bond_stereo_one_hot, encode_unknown=True),  # 7
        ])
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def build(self, smiles: str):
        d_atom_feats = 138
        d_bond_feats = 14
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
        mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
        n_atoms = mol.GetNumAtoms()
        atom_features = []

        for atom_id in range(n_atoms):
            atom = mol.GetAtomWithIdx(atom_id)
            atom_features.append(self.atom_featurizer(atom))
        atomIDPair_to_tripletId = np.ones(shape=(n_atoms, n_atoms)) * np.nan
        virtual_atom_and_virtual_node_labels = []

        atom_pairs_features_in_triplets = []
        bond_features_in_triplets = []

        bonded_atoms = set()
        triplet_id = 0
        for bond in mol.GetBonds():
            begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
            atom_pairs_features_in_triplets.append([atom_features[begin_atom_id], atom_features[end_atom_id]])
            bond_feature = self.bond_featurizer(bond)
            bond_features_in_triplets.append(bond_feature)
            bonded_atoms.add(begin_atom_id)
            bonded_atoms.add(end_atom_id)
            virtual_atom_and_virtual_node_labels.append(0)
            atomIDPair_to_tripletId[begin_atom_id, end_atom_id] = atomIDPair_to_tripletId[
                end_atom_id, begin_atom_id
            ] = triplet_id
            triplet_id += 1

        for atom_id in range(n_atoms):
            if atom_id not in bonded_atoms:
                atom_pairs_features_in_triplets.append(
                    [atom_features[atom_id], [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats]
                )
                bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
                virtual_atom_and_virtual_node_labels.append(VIRTUAL_ATOM_INDICATOR)

        edges = []
        paths = []
        line_graph_path_labels = []
        mol_graph_path_labels = []
        virtual_path_labels = []
        self_loop_labels = []
        for i in range(n_atoms):
            node_ids = atomIDPair_to_tripletId[i]
            node_ids = node_ids[~np.isnan(node_ids)]
            if len(node_ids) >= 2:
                new_edges = list(permutations(node_ids, 2))
                edges.extend(new_edges)
                new_paths = [
                    [new_edge[0]] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [new_edge[1]]
                    for new_edge in new_edges
                ]
                paths.extend(new_paths)
                n_new_edges = len(new_edges)
                line_graph_path_labels.extend([1] * n_new_edges)
                mol_graph_path_labels.extend([0] * n_new_edges)
                virtual_path_labels.extend([0] * n_new_edges)
                self_loop_labels.extend([0] * n_new_edges)

        adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
        nx_g = nx.from_numpy_array(adj_matrix)
        paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g, self.max_length + 1))
        for i in paths_dict.keys():
            for j in paths_dict[i]:
                path = paths_dict[i][j]
                path_length = len(path)
                if 3 < path_length <= self.max_length + 1:
                    triplet_ids = [
                        atomIDPair_to_tripletId[path[pi], path[pi + 1]]
                        for pi in range(len(path) - 1)
                    ]
                    path_start_triplet_id = triplet_ids[0]
                    path_end_triplet_id = triplet_ids[-1]
                    triplet_path = triplet_ids[1:-1]
                    triplet_path = (
                        [path_start_triplet_id]
                        + triplet_path
                        + [VIRTUAL_PATH_INDICATOR] * (self.max_length - len(triplet_path) - 2)
                        + [path_end_triplet_id]
                    )
                    paths.append(triplet_path)
                    edges.append([path_start_triplet_id, path_end_triplet_id])
                    line_graph_path_labels.append(0)
                    mol_graph_path_labels.append(1)
                    virtual_path_labels.append(0)
                    self_loop_labels.append(0)

        for n in range(self.n_virtual_nodes):
            for i in range(len(atom_pairs_features_in_triplets) - n):
                edges.append([len(atom_pairs_features_in_triplets), i])
                edges.append([i, len(atom_pairs_features_in_triplets)])
                paths.append(
                    [len(atom_pairs_features_in_triplets)]
                    + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2)
                    + [i]
                )
                paths.append(
                    [i]
                    + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2)
                    + [len(atom_pairs_features_in_triplets)]
                )
                line_graph_path_labels.extend([0, 0])
                mol_graph_path_labels.extend([0, 0])
                virtual_path_labels.extend([n + 1, n + 1])
                self_loop_labels.extend([0, 0])
            atom_pairs_features_in_triplets.append(
                [[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats]
            )
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
            virtual_atom_and_virtual_node_labels.append(n + 1)

        if self.add_self_loop:
            for i in range(len(atom_pairs_features_in_triplets)):
                edges.append([i, i])
                paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [i])
                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(0)
                virtual_path_labels.append(0)
                self_loop_labels.append(1)
        edges = np.array(edges, dtype=np.int64)
        data = (edges[:, 0], edges[:, 1])
        g = dgl.graph(data)
        g.ndata["begin_end"] = torch.FloatTensor(atom_pairs_features_in_triplets)
        g.ndata["edge"] = torch.FloatTensor(bond_features_in_triplets)
        g.ndata["vavn"] = torch.LongTensor(virtual_atom_and_virtual_node_labels)
        g.edata["path"] = torch.LongTensor(paths)
        g.edata["lgp"] = torch.BoolTensor(line_graph_path_labels)
        g.edata["mgp"] = torch.BoolTensor(mol_graph_path_labels)
        g.edata["vp"] = torch.BoolTensor(virtual_path_labels)
        g.edata["sl"] = torch.BoolTensor(self_loop_labels)
        return g
