from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split

from .dataset import PolymerDataset


@dataclass(frozen=True)
class FoldSplit:
    repeat: int
    fold: int
    outer_seed: int
    inner_seed: int
    model_seed: int
    inner_train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray


def canonical_group_ids(groups: np.ndarray) -> np.ndarray:
    canonical = np.asarray(groups, dtype=str)
    return np.asarray([
        hashlib.blake2b(value.encode('utf-8'), digest_size=16).hexdigest()
        for value in canonical
    ])


def generate_splits(
    dataset: PolymerDataset,
    *,
    n_folds: int,
    n_trials: int,
    seed: int,
    inner_validation_size: float,
) -> list[FoldSplit]:
    indices = np.arange(len(dataset))
    groups = np.asarray(dataset.groups, dtype=str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_folds:
        raise ValueError(
            f'{dataset.properties}: {len(unique_groups)} canonical groups are insufficient '
            f'for {n_folds} outer folds'
        )

    outer_seeds = random.Random(seed).sample(range(1000), n_trials)
    splits: list[FoldSplit] = []
    for repeat, outer_seed in enumerate(outer_seeds, start=1):
        if len(unique_groups) == len(indices):
            outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=outer_seed)
            outer_splits = outer_cv.split(indices)
        else:
            outer_cv = GroupKFold(n_splits=n_folds, shuffle=True, random_state=outer_seed)
            outer_splits = outer_cv.split(indices, groups=groups)

        for fold, (outer_train_idx, test_idx) in enumerate(outer_splits, start=1):
            generated = np.random.SeedSequence([seed, repeat, fold]).generate_state(2)
            inner_seed, model_seed = (int(value) for value in generated)
            outer_train_groups = groups[outer_train_idx]

            if len(np.unique(outer_train_groups)) == len(outer_train_idx):
                inner_train_idx, validation_idx = train_test_split(
                    outer_train_idx,
                    test_size=inner_validation_size,
                    random_state=inner_seed,
                )
            else:
                inner_cv = GroupShuffleSplit(
                    n_splits=1,
                    test_size=inner_validation_size,
                    random_state=inner_seed,
                )
                inner_train_pos, validation_pos = next(
                    inner_cv.split(outer_train_idx, groups=outer_train_groups)
                )
                inner_train_idx = outer_train_idx[inner_train_pos]
                validation_idx = outer_train_idx[validation_pos]

            split = FoldSplit(
                repeat=repeat,
                fold=fold,
                outer_seed=outer_seed,
                inner_seed=inner_seed,
                model_seed=model_seed,
                inner_train_idx=np.sort(np.asarray(inner_train_idx, dtype=np.int64)),
                validation_idx=np.sort(np.asarray(validation_idx, dtype=np.int64)),
                test_idx=np.sort(np.asarray(test_idx, dtype=np.int64)),
            )
            splits.append(split)

    return splits


def build_manifest(
    dataset: PolymerDataset,
    *,
    n_folds: int,
    n_trials: int,
    seed: int,
    inner_validation_size: float,
) -> dict:
    fold_entries = []
    for split in generate_splits(
        dataset,
        n_folds=n_folds,
        n_trials=n_trials,
        seed=seed,
        inner_validation_size=inner_validation_size,
    ):
        fold_entries.append({
            'repeat': split.repeat,
            'fold': split.fold,
            'outer_seed': split.outer_seed,
            'inner_seed': split.inner_seed,
            'model_seed': split.model_seed,
            'validation_row_ids': dataset.row_indices[split.validation_idx].tolist(),
            'test_row_ids': dataset.row_indices[split.test_idx].tolist(),
        })

    return {
        'source_file': str(Path(dataset.dataset_dir) / dataset.data_file),
        'task': dataset.properties[0],
        'n_folds': n_folds,
        'n_trials': n_trials,
        'seed': seed,
        'inner_validation_size': inner_validation_size,
        'row_ids': dataset.row_indices.tolist(),
        'group_ids': canonical_group_ids(dataset.groups).tolist(),
        'folds': fold_entries,
    }


def save_manifest(path: str | Path, manifest: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'{path.name}.tmp')
    with gzip.open(temporary_path, 'wt', encoding='utf-8') as file:
        json.dump(manifest, file, separators=(',', ':'))
    os.replace(temporary_path, path)


def load_manifest(path: str | Path) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        return json.load(file)


def validate_manifest(
    manifest: dict,
    dataset: PolymerDataset,
    *,
    n_folds: int,
    n_trials: int,
    seed: int,
    inner_validation_size: float,
):
    expected = {
        'task': dataset.properties[0],
        'n_folds': n_folds,
        'n_trials': n_trials,
        'seed': seed,
        'inner_validation_size': inner_validation_size,
    }
    for key, value in expected.items():
        if manifest[key] != value:
            raise ValueError(f'Split manifest {key} does not match the requested data/settings')
    if (manifest['row_ids'] != dataset.row_indices.tolist()
            or manifest['group_ids'] != canonical_group_ids(dataset.groups).tolist()):
        raise ValueError(f'Split manifest samples do not match task {dataset.properties[0]}')


def splits_from_manifest(manifest: dict, dataset: PolymerDataset) -> list[FoldSplit]:
    positions = {int(row_id): i for i, row_id in enumerate(dataset.row_indices)}
    all_idx = np.arange(len(dataset), dtype=np.int64)
    splits = []

    for fold_entry in manifest['folds']:
        validation_idx = np.asarray(
            [positions[int(row_id)] for row_id in fold_entry['validation_row_ids']],
            dtype=np.int64,
        )
        test_idx = np.asarray(
            [positions[int(row_id)] for row_id in fold_entry['test_row_ids']],
            dtype=np.int64,
        )
        inner_train_idx = np.setdiff1d(all_idx, np.union1d(validation_idx, test_idx))
        split = FoldSplit(
            repeat=int(fold_entry['repeat']),
            fold=int(fold_entry['fold']),
            outer_seed=int(fold_entry['outer_seed']),
            inner_seed=int(fold_entry['inner_seed']),
            model_seed=int(fold_entry['model_seed']),
            inner_train_idx=inner_train_idx,
            validation_idx=np.sort(validation_idx),
            test_idx=np.sort(test_idx),
        )
        splits.append(split)

    return splits
