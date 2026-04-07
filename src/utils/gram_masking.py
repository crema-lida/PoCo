import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from sklearn.utils.validation import check_symmetric

from .utils import expand_fragment_tokens, ATOMS
from .fragment import fragment_by_atom_sets


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


def _get_atom_token_info(tokens: list[str]) -> tuple[list[int], list[bool]]:
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
    return atom_token_ids, is_atom_tok


def get_atom_groups(
    mol: Chem.Mol,
    W: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    neighbors: list[list[int]] = [[] for _ in range(mol.GetNumAtoms())]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if W[i, j] >= threshold:
            neighbors[i].append(j)
            neighbors[j].append(i)

    groups: list[list[int]] = []
    seen = [False] * mol.GetNumAtoms()
    for start in range(mol.GetNumAtoms()):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        group = []
        while stack:
            u = stack.pop()
            group.append(u)
            for v in neighbors[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        groups.append(sorted(group))
    return groups


def make_gram_fragments(
    smiles: str,
    tokens: list[str],
    embeds: torch.Tensor,
    topk: int | None = None,
    score_threshold: float = 0.6,
    w_single: float = 0.0,
    w_double: float = 0.6,
    w_triple: float = 0.6,
    w_aromatic: float = 0.6,
) -> dict:
    # Choose fragments by thresholding bond scores.
    atom_token_ids, is_atom_tok = _get_atom_token_info(tokens)
    mol = Chem.MolFromSmiles(smiles)

    # Build atom-level similarity matrix from embeddings and bond boosts.
    W = build_similarity(
        embeds[is_atom_tok],
        mol,
        topk=topk,
        w_single=w_single,
        w_double=w_double,
        w_triple=w_triple,
        w_aromatic=w_aromatic,
    )

    atom_groups = get_atom_groups(mol, W, score_threshold)
    frags = fragment_by_atom_sets(mol, atom_groups)

    star_idx = [i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() == '*']
    del_idx = []
    for i, blocks in enumerate(atom_groups):
        for s in star_idx:
            if s in blocks:
                blocks.remove(s)
        if len(blocks) == 0:
            del_idx.append(i)
    for i in reversed(del_idx):
        del atom_groups[i]
        del frags[i]

    frag_ids = []
    for blocks in atom_groups:
        atom_tok = set()
        for atom_idx in blocks:
            atom_tok.add(atom_token_ids[atom_idx])
        full = expand_fragment_tokens(tokens, is_atom_tok, atom_tok)
        frag_ids.append(full)

    return {
        'atom_groups': atom_groups,
        'frag_ids': frag_ids,
        'fragments': frags,
    }
