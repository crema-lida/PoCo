from __future__ import annotations
import hashlib
import json
import os
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
        dataset_dir: str,
        data_file: str,
        smiles_column: str,
        categories: dict[str, list[str]],
        log10_col: list[str] | None,
        logm1_col: list[str] | None,
        data: pd.DataFrame,
        canonical_smiles: np.ndarray,
        embeddings: np.ndarray,
        encoder_path: str,
        pooling: str,
        concat_last_layers: int | None,
        embedding_cache: str | None,
        properties: list[str] | None = None,
        row_indices: list[int] | np.ndarray | None = None,
    ):
        self.dataset_dir = dataset_dir
        self.data_file = data_file
        self.smiles_column = smiles_column
        self.categories = {category: list(names) for category, names in categories.items()}
        self.all_properties = [name for names in self.categories.values() for name in names]
        self.log10_col = list(log10_col or [])
        self.logm1_col = list(logm1_col or [])
        self.data = data
        self.canonical_smiles = np.asarray(canonical_smiles, dtype=object)
        self.embeddings = embeddings
        self.encoder_path = encoder_path
        self.pooling = pooling
        self.concat_last_layers = concat_last_layers
        self.embedding_cache = embedding_cache

        self.properties = self.normalize_tasks(properties)
        if row_indices is None:
            self.row_universe = self.data.index.to_numpy(copy=True)
        else:
            self.row_universe = np.asarray(row_indices).copy()
        view = self.data.loc[self.row_universe, self.properties].dropna()

        self.row_indices = view.index.to_numpy(copy=True)
        self.fingerprints = self.embeddings[self.row_indices]
        self.values = view.to_numpy(copy=True)
        self.groups = self.canonical_smiles[self.row_indices]
        self.log10_idx = np.array([name in self.log10_col for name in self.properties], dtype=bool)
        self.logm1_idx = np.array([name in self.logm1_col for name in self.properties], dtype=bool)

    @classmethod
    def from_dir(
        cls,
        dataset_dir: str,
        encoder_path: str,
        pooling: str = 'mean',
        concat_last_layers: int | None = None,
        embedding_cache: str | None = None,
        **kwargs,
    ) -> PolymerDataset:
        dataset_path = Path(dataset_dir)
        config_path = dataset_path / 'dataset.yaml'
        if not config_path.is_file():
            raise FileNotFoundError(f'Dataset config not found: {config_path}')

        with config_path.open() as f:
            config = yaml.safe_load(f) or {}

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
        canonical_smiles = np.asarray(parallel_canonicalize(smiles.to_numpy()), dtype=object)
        cache_metadata = {
            'canonical_smiles_sha256': cls._canonical_smiles_sha256(canonical_smiles),
            'encoder_path': encoder_path,
            'pooling': pooling,
            'concat_last_layers': concat_last_layers,
            'encode_kwargs': {key: value for key, value in kwargs.items()
                              if key not in ('batch_size', 'num_workers')},
            'n_rows': len(canonical_smiles),
        }
        embeddings = None
        if embedding_cache is not None:
            embeddings = cls._load_embedding_cache(embedding_cache, cache_metadata)
            if embeddings is not None:
                print(f'Loaded cached embeddings from {embedding_cache}.')
        if embeddings is None:
            encode = get_encoder(encoder_path)
            print('Encoding SMILES strings...')
            embeddings = np.asarray(encode(
                canonical_smiles.tolist(),
                pooling=pooling,
                concat_last_layers=concat_last_layers,
                **kwargs,
            ))
            if embedding_cache is not None:
                cls._save_embedding_cache(
                    embedding_cache,
                    embeddings,
                    cache_metadata,
                )
                print(f'Cached embeddings at {embedding_cache}.')

        return cls(
            dataset_dir=str(dataset_path),
            data_file=data_file,
            smiles_column=smiles_column,
            categories=categories,
            log10_col=log10_col,
            logm1_col=logm1_col,
            data=data,
            canonical_smiles=canonical_smiles,
            embeddings=embeddings,
            encoder_path=encoder_path,
            pooling=pooling,
            concat_last_layers=concat_last_layers,
            embedding_cache=embedding_cache,
        )

    def subset(
        self,
        indices: list[int] | np.ndarray | None = None,
        tasks: list[str] | None = None,
    ) -> PolymerDataset:
        selected_tasks = self.normalize_tasks(self.properties if tasks is None else tasks)
        eligible_rows = (
            self.data.loc[self.row_universe, selected_tasks]
            .dropna()
            .index.to_numpy(copy=True)
        )
        if indices is not None:
            selected_rows = eligible_rows[np.asarray(indices)]
        else:
            selected_rows = self.row_universe

        return type(self)(
            dataset_dir=self.dataset_dir,
            data_file=self.data_file,
            smiles_column=self.smiles_column,
            categories=self.categories,
            log10_col=self.log10_col,
            logm1_col=self.logm1_col,
            data=self.data,
            canonical_smiles=self.canonical_smiles,
            embeddings=self.embeddings,
            encoder_path=self.encoder_path,
            pooling=self.pooling,
            concat_last_layers=self.concat_last_layers,
            embedding_cache=self.embedding_cache,
            properties=selected_tasks,
            row_indices=selected_rows,
        )

    def to_config_dict(self) -> dict[str, Any]:
        return {
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
                'embedding_cache': self.embedding_cache,
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

    @staticmethod
    def _canonical_smiles_sha256(canonical_smiles: np.ndarray) -> str:
        digest = hashlib.sha256()
        for value in canonical_smiles:
            encoded = str(value).encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'little'))
            digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def _load_embedding_cache(
        path: str,
        expected_metadata: dict,
    ) -> np.ndarray | None:
        metadata_path = Path(f'{path}.json')
        if not Path(path).is_file() or not metadata_path.is_file():
            return None
        with metadata_path.open(encoding='utf-8') as file:
            metadata = json.load(file)
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            return None
        return np.load(path, mmap_mode='r')

    @staticmethod
    def _save_embedding_cache(path: str, embeddings: np.ndarray, metadata: dict):
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        array_tmp = cache_path.with_name(f'{cache_path.name}.tmp')
        with array_tmp.open('wb') as file:
            np.save(file, embeddings)
        metadata_path = Path(f'{path}.json')
        metadata_path.unlink(missing_ok=True)
        os.replace(array_tmp, cache_path)
        metadata_tmp = metadata_path.with_name(f'{metadata_path.name}.tmp')
        with metadata_tmp.open('w', encoding='utf-8') as file:
            json.dump(metadata, file, indent=2)
        os.replace(metadata_tmp, metadata_path)
