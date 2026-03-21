import pandas as pd
import numpy as np

import sys
sys.path.append('src/')

from benchmark import get_encoder
from utils import parallel_canonicalize


class PolymerDataset:
    categories: dict[str, str] = {}
    properties: list[str] = []
    log10_col = []
    logm1_col = []
    data: pd.DataFrame | None = None
    embeddings: np.ndarray | None = None
    col_name = 'smiles'

    @classmethod
    def encode_smiles(cls, data: pd.DataFrame, encoder_path: str, **kwargs):
        cls.properties = [name for names in cls.categories.values() for name in names]
        cls.data = data[cls.properties]
        smiles = data[cls.col_name]
        print(f'Canonicalizing {len(smiles)} SMILES strings...')
        smiles = parallel_canonicalize(smiles.to_numpy()).tolist()
        encode = get_encoder(encoder_path)
        print('Encoding SMILES strings...')
        cls.embeddings = encode(smiles, **kwargs)
        print('Done.')

    def __init__(self, indices=None):
        data = self.data[self.properties].dropna()
        if indices is not None:
            data = data.iloc[indices]

        self.log10_idx = np.array([s in self.log10_col for s in self.properties])
        self.logm1_idx = np.array([s in self.logm1_col for s in self.properties])
        
        self.fingerprints = self.embeddings[data.index]
        self.values = data.to_numpy()

    def __len__(self):
        return len(self.fingerprints)
    
    def __getitem__(self, idx):
        return self.fingerprints[idx], self.values[idx]


class MTL(PolymerDataset):
    categories = {
        'Thermodynamic & physical': ['Eat', 'Xc'],
        'Electronic': ['Egc', 'Egb', 'Eea', 'Ei'],
        'Optical & dielectric': ['nc', 'eps'],
    }

    @classmethod
    def encode_smiles(cls, filename, encoder_path, **kwargs):
        data = pd.read_csv(filename)
        super().encode_smiles(data, encoder_path, **kwargs)


class RadonPy(PolymerDataset):
    categories = {
        'Thermal': ['thermal_conductivity', 'thermal_diffusivity', 'linear_expansion', 'volume_expansion'],
        'Thermodynamic & physical': ['density', 'Rg', 'self-diffusion', 'Cp', 'Cv'],
        'Electronic': ['qm_homo_monomer', 'qm_lumo_monomer', 'qm_dipole_monomer', 'qm_polarizability_monomer'],
        'Optical & dielectric': ['static_dielectric_const', 'refractive_index'],
        'Mechanical': ['compressibility', 'isentropic_compressibility', 'bulk_modulus', 'isentropic_bulk_modulus'],
    }
    log10_col = ['self-diffusion', 'static_dielectric_const']

    @classmethod
    def encode_smiles(cls, filename, encoder_path, **kwargs):
        data = pd.read_csv(filename)
        super().encode_smiles(data, encoder_path, **kwargs)


class PolyOmics(PolymerDataset):
    categories = {
        'Thermal': ['thermal_conductivity', 'thermal_diffusivity', 'CLTE', 'tg'],
        'Thermodynamic & physical': ['density', 'Rg', 'self-diffusion', 'Cp', 'Cv'],
        'Electronic': ['qm_homo_monomer1', 'qm_lumo_monomer1', 'qm_dipole_monomer1', 'qm_polarizability_monomer1'],
        'Optical & dielectric': ['static_dielectric_const', 'refractive_index'],
        'Mechanical': ['compressibility', 'isentropic_compressibility', 'bulk_modulus', 'isentropic_bulk_modulus'],
    }
    col_name = 'smiles_list'

    @classmethod
    def encode_smiles(cls, filename, encoder_path, **kwargs):
        data = pd.read_csv(filename)
        super().encode_smiles(data, encoder_path, **kwargs)
