from mordred import Calculator, descriptors
import numpy as np
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem

_GLOBAL_CALC = Calculator(descriptors, ignore_3D=True)
_MORGAN_GEN = GetMorganGenerator(radius=4, fpSize=1024)


def _fp_md_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, None, None

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    ec_fp = _MORGAN_GEN.GetFingerprint(mol)
    fp = list(map(int, list(maccs_fp + ec_fp)))

    des = np.array(list(_GLOBAL_CALC(mol).values()), dtype=np.float32)
    des = np.where(np.isnan(des), 0, des)
    des = np.where(des > 1e12, 1e12, des)
    return smiles, fp, des
