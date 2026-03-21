from __future__ import annotations

from typing import List, Tuple
import numpy as np


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError(
            "MMPolymer 3D encoder requires rdkit to generate coordinates."
        ) from exc
    return Chem, AllChem


def smi2_2dcoords(smiles: str) -> np.ndarray:
    Chem, AllChem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coords = mol.GetConformer().GetPositions().astype(np.float32)
    if len(mol.GetAtoms()) != len(coords):
        raise ValueError(f"2D coordinates shape is not aligned with {smiles}")
    return coords


def smi2_3dcoords(smiles: str, count: int) -> List[np.ndarray]:
    Chem, AllChem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = AllChem.AddHs(mol)

    coords_list = []
    for seed in range(count):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                    coords = mol.GetConformer().GetPositions()
                except Exception:
                    coords = smi2_2dcoords(smiles)
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smiles)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)
                    coords = mol_tmp.GetConformer().GetPositions()
                except Exception:
                    coords = smi2_2dcoords(smiles)
            else:
                coords = smi2_2dcoords(smiles)
        except Exception:
            coords = smi2_2dcoords(smiles)

        if len(mol.GetAtoms()) != len(coords):
            raise ValueError(f"3D coordinates shape is not aligned with {smiles}")
        coords_list.append(coords.astype(np.float32))
    return coords_list


def _prepare_star_smi(smiles: str) -> Tuple[str, List[int]]:
    Chem, _ = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    star_atoms_id = [idx for idx, atom_symbol in enumerate(atoms) if atom_symbol == "*"]
    if len(star_atoms_id) != 2:
        raise ValueError(f"Expected 2 '*' atoms in SMILES: {smiles}")

    star_pair_list: List[int] = []
    for star_id in star_atoms_id:
        star_pair_list.append(star_id)
        star_atom = mol.GetAtomWithIdx(star_id)
        neighbors = star_atom.GetNeighbors()
        if len(neighbors) != 1:
            raise ValueError(f"Unexpected '*' neighbors in SMILES: {smiles}")
        star_pair_list.append(neighbors[0].GetIdx())

    pair_1_star = star_pair_list[0]
    pair_1 = star_pair_list[3]
    mol.GetAtomWithIdx(pair_1_star).SetAtomicNum(
        mol.GetAtomWithIdx(pair_1).GetAtomicNum()
    )

    pair_2_star = star_pair_list[2]
    pair_2 = star_pair_list[1]
    mol.GetAtomWithIdx(pair_2_star).SetAtomicNum(
        mol.GetAtomWithIdx(pair_2).GetAtomicNum()
    )

    return Chem.MolToSmiles(mol), star_atoms_id


def prepare_mm_polymer_conformers(
    smiles: str,
    conf_size: int = 1,
    max_atoms_2d: int = 400,
) -> Tuple[List[str], List[np.ndarray]]:
    Chem, AllChem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    star_atoms_id = None
    smi_for_coords = smiles
    if "*" in smiles:
        smi_for_coords, star_atoms_id = _prepare_star_smi(smiles)

    if mol.GetNumAtoms() > max_atoms_2d:
        coord_list = [smi2_2dcoords(smi_for_coords)] * conf_size
    else:
        coord_list = smi2_3dcoords(smi_for_coords, conf_size)

    mol = Chem.MolFromSmiles(smi_for_coords)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

    if star_atoms_id is not None:
        for idx in star_atoms_id:
            if idx < len(atoms):
                atoms[idx] = "*"

    return atoms, coord_list
