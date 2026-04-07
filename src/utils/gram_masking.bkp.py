import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from sklearn.cluster import SpectralClustering
from sklearn.utils.validation import check_symmetric

from .utils import expand_fragment_tokens, ATOMS
from .fragment import merge_and_fragment, fragment_by_atom_sets


def build_similarity(
    x: torch.Tensor,
    mol: Chem.Mol,
    *,
    topk: int | None = None,
    w_single: float = 0.15,
    w_double: float = 0.8,
    w_triple: float = 0.8,
    w_aromatic: float = 0.8,
) -> np.ndarray:
    x = F.normalize(x, dim=-1)
    W = (x @ x.T).clamp(min=0).numpy()
    L = W.shape[0]

    # Keep top-k per row (with symmetric retention).
    if topk is not None and topk < L:
        mask = np.zeros_like(W, dtype=bool)
        idx = np.argpartition(-W, kth=min(topk, L - 1), axis=1)[:, :topk]
        mask[np.arange(L)[:, None], idx] = True
        W = np.where(mask | mask.T, W, 0.)

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()

        # Bond type selection.
        bt = b.GetBondType()
        match bt:
            case Chem.BondType.DOUBLE:
                w = w_double
            case Chem.BondType.TRIPLE:
                w = w_triple
            case Chem.BondType.AROMATIC:
                w = w_aromatic
            case _:
                w = w_single

        # Enhance bond connection
        W[i, j] = min(W[i, j] + w, 1)
        W[j, i] = min(W[j, i] + w, 1)

    return check_symmetric(W, raise_warning=False)


def auto_select_K_eigengap(W: np.ndarray, K_min: int = 2, K_max: int = 8) -> int:
    # Symmetrize.
    W = 0.5 * (W + W.T)
    n = W.shape[0]
    if n <= 2:
        return n

    d = W.sum(axis=1)
    d[d == 0] = 1e-8
    Dm12 = np.diag(1.0 / np.sqrt(d))
    L = np.eye(n) - Dm12 @ W @ Dm12

    vals, _ = np.linalg.eigh(L)        # Ascending order.
    vals = np.clip(vals, 0.0, None)

    # We can use at most n-1 gaps.
    m = int(min(len(vals) - 1, K_max))
    if m < K_min:
        # Fallback to the largest feasible K in [2, m], or 2.
        return max(2, min(m, K_min))

    gaps = vals[1:m+1] - vals[0:m]     # Length = m.
    search = gaps[K_min-1 : m]          # Upper bound is m, includes index m-1.

    if search.size == 0 or np.allclose(search, 0.0):
        # Fallback when the eigengap is not clear.
        return min(max(2, K_min), m)

    idx = int(np.argmax(search))        # 0-based in 'search'
    k = (K_min - 1) + idx + 1           # Map back to gap index +1 => K.
    k = int(max(2, min(k, m)))
    return k


def get_atom_clusters(labels) -> dict[int, list[list[int]]]:
    blocks = {k: [] for k in np.unique(labels).tolist()}
    s = 0
    for i in range(1, len(labels) + 1):
        k = labels[i - 1]
        if i == len(labels) or labels[i] != k:
            blocks[k].append(list(range(s, i)))
            s = i
    return blocks


def make_gram_fragments(
    smiles: str,
    tokens: list[str],
    embeds: torch.Tensor,
    topk: int | None = None,
    w_single: float = 0.15,
    w_double: float = 0.8,
    w_triple: float = 0.8,
    w_aromatic: float = 0.8,
    n_clusters: int | None = None,
    k_max: int = 8,
) -> dict:
    # Map each atom to the token position it originates from.
    atom_token_ids = []
    is_atom_tok = [False] * len(tokens)
    in_bracket = False
    for i, tok in enumerate(tokens):
        if tok == '[':
            in_bracket = True
            atom_token_ids.append(i + 1)
            is_atom_tok[i + 1] = True
        elif tok == ']':
            in_bracket = False
        elif not in_bracket and tok in ATOMS:
            atom_token_ids.append(i)
            is_atom_tok[i] = True

    mol = Chem.MolFromSmiles(smiles)
    # Build atom-level similarity matrix from embeddings and bond boosts.
    W = build_similarity(embeds[is_atom_tok], mol, topk=topk,
                         w_single=w_single, w_double=w_double,
                         w_triple=w_triple, w_aromatic=w_aromatic)

    if n_clusters is None:
        # Choose cluster count based on eigengap if not fixed.
        n_clusters = auto_select_K_eigengap(
            W,
            K_min=2,
            K_max=min(k_max, max(2, len(atom_token_ids) // 3)),
        )
    labels = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', n_init=10
    ).fit_predict(W)
    
    atom_clusters = get_atom_clusters(labels)

    star_idx = [i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() == '*']
    
    merged_blocks, frags = [], []
    for blocks in atom_clusters.values():
        # Merge connected clusters and fragment the molecule accordingly.
        mb, fs = merge_and_fragment(mol, blocks)
        merged_blocks.extend(mb)
        frags.extend(fs)

    # Drop dummy [*] atoms that can appear after fragmentation.
    del_idx = []
    for i, blocks in enumerate(merged_blocks):
        for s in star_idx:
            if s in blocks:
                blocks.remove(s)
        if len(blocks) == 0:
            del_idx.append(i)
    for i in reversed(del_idx):
        del merged_blocks[i]
        del frags[i]

    frag_ids = []
    for blocks in merged_blocks:
        atom_tok = set()
        for atom_idx in blocks:
            atom_tok.add(atom_token_ids[atom_idx])
        # Expand atom token indices to include bracket/context tokens.
        full = expand_fragment_tokens(tokens, is_atom_tok, atom_tok)
        frag_ids.append(full)

    return {
        'atom_groups': merged_blocks,
        'frag_ids': frag_ids,
        'fragments': frags,
    }
