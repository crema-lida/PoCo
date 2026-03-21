import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ELEMENTS = {'H', 'B', 'C', 'N', 'O', 'Si', 'P', 'S', 'F', 'Cl', 'Br', 'I'}
AROMATIC = {'b', 'c', 'n', 'o', 'p', 's'}
ATOMS = ELEMENTS | AROMATIC | {'[*]', '[UNK]'}


def _apply_chunk(args):
    chunk, func = args
    return np.fromiter((func(x) for x in chunk), dtype=object, count=len(chunk))


def parallel_apply(arr, func):
    n_jobs = max(1, (os.cpu_count() or 1) - 1)
    chunks = np.array_split(arr, n_jobs)
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        parts = list(ex.map(_apply_chunk, [(c, func) for c in chunks]))
    return np.concatenate(parts)


def bracket_pairs(tokens):
    brackets = {'(': [], '[': []}
    bracket_map = {')': '(', ']': '['}
    pairs = []
    for i, t in enumerate(tokens):
        if t in brackets:
            brackets[t].append(i)
        elif t in bracket_map:
            start = brackets[bracket_map[t]].pop()
            pairs.append((start, i))
    return pairs

def expand_fragment_tokens(tokens, is_atom_tok, atom_tok_set):
    """
    Include additional tokens (bonds, parentheses, ring numbers) to fragment token set.
    """
    frag_tokens = set(atom_tok_set)

    def left_atom(i):
        j = i - 1
        while j >= 0 and not is_atom_tok[j]:
            j -= 1
        return j

    def right_atom(i):
        j = i + 1
        while j < len(tokens) and not is_atom_tok[j]:
            j += 1
        return j

    bonds  = {'=', '#', '/', '\\', '@'}
    digits = set('0123456789')

    for i, t in enumerate(tokens):
        # include bonds connected to fragment atoms
        if t in bonds:
            if left_atom(i) in frag_tokens or right_atom(i) in frag_tokens:
                frag_tokens.add(i)
        
        # include ring numbers connected to fragment atoms
        elif t in digits:
            if left_atom(i) in frag_tokens:
                frag_tokens.add(i)
                if i - 1 >= 0 and tokens[i - 1] == '%':
                    frag_tokens.add(i - 1)

    # include parentheses if all enclosed atom tokens are in the fragment
    for l, r in bracket_pairs(tokens):
        if all(j in frag_tokens for j in range(l + 1, r) if is_atom_tok[j]):
            frag_tokens.update(range(l, r + 1))

    return sorted(frag_tokens)
