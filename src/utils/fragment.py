import numpy as np
from rdkit import Chem


def get_connected_components(adj: np.ndarray) -> list[set[int]]:
    """
    Returns the connected components of the adjacency matrix.
    """
    K = adj.shape[0]
    seen = [False] * K
    comps = []
    for s in range(K):
        if seen[s]:
            continue
        comp = {s}
        q = [s]
        seen[s] = True
        while q:
            u = q.pop()
            nbrs, = adj[u].nonzero()
            for v in nbrs.tolist():
                if not seen[v]:
                    seen[v] = True
                    comp.add(v)
                    q.append(v)
        comps.append(comp)
    return comps


def build_adjacency_list(mol: Chem.Mol) -> list[list[int]]:
    neighbors: list[list[int]] = [[] for _ in range(mol.GetNumAtoms())]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        neighbors[i].append(j)
        neighbors[j].append(i)
    return neighbors


def merge_atom_clusters(blocks: list[list[int]], comps: list[set[int]]) -> list[list[int]]:
    """
    Returns merged atom clusters.
    `blocks`: list of atom clusters, each is a set of atom indices.\n
    `comps`: list of connected components of cluster indices.
    """
    merged = []
    for comp in comps:
        atoms = set()
        for bi in comp:
            atoms.update(i for i in blocks[bi])
        merged.append(sorted(list(atoms)))
    return merged


def compute_cut_bonds(
    mol: Chem.Mol,
    blocks: list[list[int]],
) -> list[int]:
    """
    Returns the bond indices to be cut to separate the atom clusters.
    """
    # set cluster label for each atom
    atom_label = -np.ones(mol.GetNumAtoms(), dtype=int)
    for cid, atoms in enumerate(blocks):
        for a in atoms:
            atom_label[a] = cid

    cut_bonds = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        li, lj = atom_label[i], atom_label[j]
        if li != lj:  # atom i,j belongs to different clusters
            cut_bonds.append(b.GetIdx())
    return cut_bonds


def fragment_by_atom_sets(
    mol: Chem.Mol,
    blocks: list[list[int]],
) -> list[str]:
    """
    Returns the fragments corresponding to the atom sets.
    """
    for a in mol.GetAtoms():
        a.SetIntProp('_orig_idx', a.GetIdx())

    cut_bonds = compute_cut_bonds(mol, blocks)

    if len(cut_bonds) == 0:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return [smiles] * len(blocks)

    frag_mol = Chem.FragmentOnBonds(
        mol,
        bondIndices=cut_bonds,
        dummyLabels=[(0, 0)] * len(cut_bonds),
    )

    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

    frag_ids: list[set[int]] = []
    for frag in frags:
        s = set()
        for a in frag.GetAtoms():
            if a.HasProp('_orig_idx'):
                s.add(a.GetIntProp('_orig_idx'))
        frag_ids.append(s)

    block_sets = [set(block) for block in blocks]
    out: list[str] = []
    for block_set in block_sets:
        best_idx, best_overlap = -1, -1
        for idx, aset in enumerate(frag_ids):
            ov = len(block_set & aset)
            if ov > best_overlap:
                best_overlap = ov
                best_idx = idx
        smiles = Chem.MolToSmiles(frags[best_idx])
        out.append(smiles)
    return out


def get_block_adjacency_matrix(
    mol: Chem.Mol,
    blocks: list[list[int]],
) -> np.ndarray:
    """
    Returns the adjacency matrix between atom sets.
    """
    k = len(blocks)
    adj = np.zeros((k, k), dtype=bool)
    if k <= 1:
        return adj
    atom_label = -np.ones(mol.GetNumAtoms(), dtype=int)
    for cid, atoms in enumerate(blocks):
        for a in atoms:
            atom_label[a] = cid
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        li, lj = atom_label[i], atom_label[j]
        if li >= 0 and lj >= 0 and li != lj:
            adj[li, lj] = True
            adj[lj, li] = True
    return adj


def split_block_into_connected_components(
    neighbors: list[list[int]],
    block: list[int],
) -> list[list[int]]:
    """
    Compute connected components within a block on the graph defined by neighbors.
    Returns [set(block)] if connected; otherwise returns multiple sub-blocks.
    """
    nodes = sorted(set(int(i) for i in block))
    if not nodes:
        return []
    in_block = set(nodes)
    # BFS/DFS to find connected components.
    seen = set()
    comps: list[list[int]] = []
    for s in nodes:
        if s in seen:
            continue
        comp = set([s])
        stack = [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            for v in neighbors[u]:
                if v in in_block and v not in seen:
                    seen.add(v)
                    comp.add(v)
                    stack.append(v)
        comps.append(list(comp))
    return comps

# -------- Connectivity validation and splitting within blocks --------
def enforce_block_connectivity(
    mol: Chem.Mol,
    blocks: list[list[int]],
    drop_star_in_blocks: bool = True,
) -> list[list[int]]:
    """
    For each block: if it is disconnected in the adjacency graph, split it.
    - drop_star_in_blocks: whether to drop [*] (AtomicNum==0) indices (recommended True).
    Returns: new disjoint blocks (each connected in the graph).
    """
    neighbors = build_adjacency_list(mol)
    is_star = np.array([a.GetAtomicNum() == 0 for a in mol.GetAtoms()], dtype=bool)
    
    new_blocks: list[list[int]] = []
    for blk in blocks:
        if drop_star_in_blocks:
            nodes = [int(i) for i in blk if not is_star[i]]
        else:
            nodes = [int(i) for i in blk]
        if not nodes:
            continue
        comps = split_block_into_connected_components(neighbors, nodes)
        new_blocks.extend(comps)

    return new_blocks


def merge_and_fragment(
    mol: Chem.Mol,
    blocks: list[list[int]],
) -> tuple[list[set[int]], list[str]]:
    """
    Merge connected atom clusters in `blocks` and fragment the molecule accordingly.
    """
    blocks = enforce_block_connectivity(mol, blocks, drop_star_in_blocks=True)
    adj = get_block_adjacency_matrix(mol, blocks)
    comps = get_connected_components(adj)
    merged_blocks = merge_atom_clusters(blocks, comps)
    frags = fragment_by_atom_sets(mol, merged_blocks)
    return merged_blocks, frags
