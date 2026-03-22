from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import sys
sys.path.append('src/')

from benchmark import get_encoder
from utils import parallel_canonicalize


class PolymerDataset:
    def __init__(
        self,
        *,
        name: str,
        dataset_dir: str,
        data_file: str,
        smiles_column: str,
        categories: dict[str, list[str]],
        log10_col: list[str] | None,
        logm1_col: list[str] | None,
        data: pd.DataFrame,
        embeddings: np.ndarray,
        encoder_path: str,
        pooling: str,
        concat_last_layers: int | None,
        properties: list[str] | None = None,
        indices: list[int] | np.ndarray | None = None,
    ):
        self.name = name
        self.dataset_dir = dataset_dir
        self.data_file = data_file
        self.smiles_column = smiles_column
        self.categories = {category: list(names) for category, names in categories.items()}
        self.all_properties = [name for names in self.categories.values() for name in names]
        self.log10_col = list(log10_col or [])
        self.logm1_col = list(logm1_col or [])
        self.data = data
        self.embeddings = embeddings
        self.encoder_path = encoder_path
        self.pooling = pooling
        self.concat_last_layers = concat_last_layers

        self.properties = self.normalize_tasks(properties)
        view = self.data[self.properties].dropna()
        if indices is not None:
            view = view.iloc[indices]

        self.row_indices = view.index.to_numpy(copy=True)
        self.fingerprints = self.embeddings[self.row_indices]
        self.values = view.to_numpy()
        self.log10_idx = np.array([name in self.log10_col for name in self.properties], dtype=bool)
        self.logm1_idx = np.array([name in self.logm1_col for name in self.properties], dtype=bool)

    @classmethod
    def from_dir(
        cls,
        dataset_dir: str,
        encoder_path: str,
        pooling: str = 'mean',
        concat_last_layers: int | None = None,
        **kwargs,
    ) -> PolymerDataset:
        dataset_path = Path(dataset_dir)
        config_path = dataset_path / 'dataset.yaml'
        if not config_path.is_file():
            raise FileNotFoundError(f'Dataset config not found: {config_path}')

        with config_path.open() as f:
            config = yaml.safe_load(f) or {}

        name = config['name']
        data_file = config['data_file']
        smiles_column = config['smiles_column']
        categories = {
            category: list(names)
            for category, names in config['categories'].items()
        }
        properties = [name for names in categories.values() for name in names]

        transforms = config.get('transforms') or {}
        log10_col = list(transforms.get('log10', []))
        logm1_col = list(transforms.get('logm1', []))

        data_path = dataset_path / data_file
        if not data_path.is_file():
            raise FileNotFoundError(f'Dataset CSV not found: {data_path}')

        data = pd.read_csv(data_path)
        cls._validate_columns(data, smiles_column, properties, data_path, config_path)

        smiles = data[smiles_column]
        print(f'Canonicalizing {len(smiles)} SMILES strings...')
        smiles = list(parallel_canonicalize(smiles.to_numpy()))
        encode = get_encoder(encoder_path)
        print('Encoding SMILES strings...')
        embeddings = encode(
            smiles,
            pooling=pooling,
            concat_last_layers=concat_last_layers,
            **kwargs,
        )
        print('Done.')

        return cls(
            name=name,
            dataset_dir=str(dataset_path),
            data_file=data_file,
            smiles_column=smiles_column,
            categories=categories,
            log10_col=log10_col,
            logm1_col=logm1_col,
            data=data,
            embeddings=embeddings,
            encoder_path=encoder_path,
            pooling=pooling,
            concat_last_layers=concat_last_layers,
        )

    def subset(
        self,
        indices: list[int] | np.ndarray | None = None,
        tasks: list[str] | None = None,
    ) -> PolymerDataset:
        selected_tasks = self.properties if tasks is None else tasks
        return type(self)(
            name=self.name,
            dataset_dir=self.dataset_dir,
            data_file=self.data_file,
            smiles_column=self.smiles_column,
            categories=self.categories,
            log10_col=self.log10_col,
            logm1_col=self.logm1_col,
            data=self.data,
            embeddings=self.embeddings,
            encoder_path=self.encoder_path,
            pooling=self.pooling,
            concat_last_layers=self.concat_last_layers,
            properties=self.normalize_tasks(selected_tasks),
            indices=indices,
        )

    def to_config_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'dataset_dir': self.dataset_dir,
            'data_file': self.data_file,
            'smiles_column': self.smiles_column,
            'categories': self.categories,
            'properties': list(self.all_properties),
            'transforms': {
                'log10': list(self.log10_col),
                'logm1': list(self.logm1_col),
            },
            'encoder': {
                'path': self.encoder_path,
                'pooling': self.pooling,
                'concat_last_layers': self.concat_last_layers,
            },
        }

    def normalize_tasks(self, tasks: list[str] | None) -> list[str]:
        if tasks is None:
            return list(self.all_properties)

        return list(tasks)

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        return self.fingerprints[idx], self.values[idx]

    @staticmethod
    def _validate_columns(
        data: pd.DataFrame,
        smiles_column: str,
        properties: list[str],
        data_path: Path,
        config_path: Path,
    ):
        missing = [column for column in [smiles_column, *properties] if column not in data.columns]
        if missing:
            raise ValueError(
                f'Missing columns {missing} in dataset file {data_path} referenced by {config_path}'
            )
