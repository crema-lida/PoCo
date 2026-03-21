from rdkit import Chem
from rdkit.Chem import BRICS

from .utils import expand_fragment_tokens, ATOMS


def make_brics_fragments(smiles: str, tokens: list[str]) -> dict:
    # get token ids of each atom in the SMILES string
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

    # make BRICS fragments
    for a in mol.GetAtoms():
        if a.GetSymbol() == '*':  # set '*' atom map number to 0
            a.SetAtomMapNum(0)
        else:  # let atom map number start from 1
            a.SetAtomMapNum(a.GetIdx() + 1)
    fragged = BRICS.BreakBRICSBonds(mol)
    frag_atom_ids = []
    mol_frags = Chem.GetMolFrags(fragged, asMols=True, fragsMolAtomMapping=frag_atom_ids)

    frag_atoms = []
    for atom_ids in frag_atom_ids:
        aset = set()
        for idx in atom_ids:
            amap = fragged.GetAtomWithIdx(idx).GetAtomMapNum()
            if amap > 0:  # ignore dummy '*'
                aset.add(amap - 1)
        frag_atoms.append(sorted(list(aset)))

    # get token ids for each fragment
    frag_ids = []
    for aset in frag_atoms:
        atom_tok = set()
        for atom_idx in aset:
            atom_tok.add(atom_token_ids[atom_idx])
        if atom_tok:
            full = expand_fragment_tokens(tokens, is_atom_tok, atom_tok)
            frag_ids.append(full)

    # get fragment SMILES
    fragments = []
    for frag in mol_frags:
        for a in frag.GetAtoms():
            a.SetAtomMapNum(0)
            a.SetIsotope(0)
        fragments.append(
            Chem.MolToSmiles(frag, isomericSmiles=False)
        )

    return {
        'atom_groups': frag_atoms,
        'frag_ids': frag_ids,
        'fragments': fragments,
    }
