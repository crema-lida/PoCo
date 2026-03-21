from copy import deepcopy

import numpy as np
from rdkit import Chem


def generate_multimer_smiles(num_repeat_units, smiles, replace_dummy_atoms=True):
    if num_repeat_units == 1:
        return smiles
    monomer = Chem.MolFromSmiles(smiles)
    repeat_points = []
    cnct_points = []
    bond_type = []
    num_dummy_atoms = 0
    for atom in monomer.GetAtoms():
        if atom.GetSymbol() == "*":
            repeat_points.append(atom.GetIdx())
            neis = atom.GetNeighbors()
            assert len(neis) == 1, f"*atom has more than one neighbor: {smiles}"
            cnct_points.append(atom.GetNeighbors()[0].GetIdx())
            bond_type.append(monomer.GetBondBetweenAtoms(atom.GetIdx(), atom.GetNeighbors()[0].GetIdx()).GetBondType())
            num_dummy_atoms += 1

    assert num_dummy_atoms == 2, "molecule has more than 2 *atoms"
    assert bond_type[0] == bond_type[1], "bond type of 2 *atoms are not same"

    num_atoms = monomer.GetNumAtoms()
    oligomer = deepcopy(monomer)
    for _ in range(num_repeat_units - 1):
        oligomer = Chem.CombineMols(oligomer, monomer)

    repeat_list = np.zeros([num_repeat_units, 2])
    cnct_list = np.zeros([num_repeat_units, 2])
    for i in range(num_repeat_units):
        repeat_list[i] = np.array(repeat_points) + i * num_atoms
        cnct_list[i] = np.array(cnct_points) + i * num_atoms

    ed_oligomer = Chem.EditableMol(oligomer)
    removed_atoms_idx = []
    for i in range(num_repeat_units - 1):
        ed_oligomer.AddBond(int(cnct_list[i, 1]), int(cnct_list[i + 1, 0]), order=bond_type[0])
        removed_atoms_idx.extend([int(repeat_list[i, 1]), int(repeat_list[i + 1, 0])])

    if replace_dummy_atoms:
        ed_oligomer.ReplaceAtom(int(repeat_list[0, 0]), Chem.Atom(1))
        ed_oligomer.ReplaceAtom(int(repeat_list[num_repeat_units - 1, 1]), Chem.Atom(1))

    for i in sorted(removed_atoms_idx, reverse=True):
        ed_oligomer.RemoveAtom(i)

    final_mol = ed_oligomer.GetMol()
    final_mol = Chem.RemoveHs(final_mol)

    return Chem.MolToSmiles(final_mol)
