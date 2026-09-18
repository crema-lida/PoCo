from dataclasses import dataclass
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules import MLP, Transform
from .dataset import PolymerDataset
from .evaluate import evaluate
from .splits import (
    build_manifest,
    canonical_group_ids,
    load_manifest,
    save_manifest,
    splits_from_manifest,
    validate_manifest,
)
from .train import train


@dataclass
class MultiTaskTrainer:
    model_config: dict
    dataset: PolymerDataset
    experiment_name: str
    tasks: list[str] | None = None
    output_dir: str = './checkpoints'
    logging_dir: str | None = None
    device: str = 'cuda'
    n_folds: int = 5
    n_trials: int = 1
    seed: int = 42
    max_epochs: int = 1
    min_steps: int = 50
    early_stopping_patience: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0
    train_batch_size: int = 64
    eval_batch_size: int = 1024
    inner_validation_size: float = 0.2
    validation_only: bool = False
    resume: bool = True
    num_workers: int = 0
    drop_last: bool = False

    def __post_init__(self):
        self.tasks = self.dataset.normalize_tasks(self.tasks)
        self.results_dir = Path(self.output_dir) / 'results' / self.experiment_name
        self.writer: SummaryWriter | None = None
        self.fold_records: list[dict] = []

    def get_config(self) -> dict:
        return {
            'model_config': self.model_config,
            'dataset': self.dataset.to_config_dict(),
            'tasks': self.tasks,
            'trainer_config': {
                'output_dir': self.output_dir,
                'experiment_name': self.experiment_name,
                'logging_dir': self.logging_dir,
                'device': self.device,
                'n_folds': self.n_folds,
                'n_trials': self.n_trials,
                'seed': self.seed,
                'max_epochs': self.max_epochs,
                'min_steps': self.min_steps,
                'early_stopping_patience': self.early_stopping_patience,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'train_batch_size': self.train_batch_size,
                'eval_batch_size': self.eval_batch_size,
                'inner_validation_size': self.inner_validation_size,
                'validation_only': self.validation_only,
                'resume': self.resume,
                'num_workers': self.num_workers,
                'drop_last': self.drop_last,
            },
        }

    def save_checkpoint(self, filepath, model, transform):
        checkpoint = {
            'model': model.state_dict(),
            'transform': transform.state_dict(),
        }
        temporary_path = f'{filepath}.tmp'
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, filepath)

    @staticmethod
    def _resume_settings(config: dict) -> dict:
        return {
            'validation_only': config['trainer_config'].get('validation_only', False),
            'model': config['model_config'],
            'dataset': {key: config['dataset'][key] for key in (
                'dataset_dir', 'data_file', 'smiles_column', 'transforms',
            )},
            'encoder': {key: config['dataset']['encoder'][key] for key in (
                'path', 'pooling', 'concat_last_layers',
            )},
            'training': {key: config['trainer_config'][key] for key in (
                'n_folds', 'n_trials', 'seed', 'max_epochs', 'min_steps', 'early_stopping_patience',
                'learning_rate', 'weight_decay', 'train_batch_size',
                'inner_validation_size', 'drop_last',
            )},
        }

    @staticmethod
    def seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def _fold_artifact_paths(task_dir: Path, repeat: int, fold: int) -> dict[str, Path]:
        name = f'fold-{repeat}.{fold}'
        return {
            'checkpoint': task_dir / f'{name}.pth',
            'metrics': task_dir / f'{name}.json',
            'predictions': task_dir / f'{name}.predictions.csv.gz',
        }

    def _load_completed_fold(self, paths: dict[str, Path]) -> bool:
        if not all(path.is_file() for path in paths.values()):
            return False
        with paths['metrics'].open(encoding='utf-8') as file:
            record = json.load(file)
        self.fold_records.append(record)
        return True

    def _save_fold_artifacts(
        self,
        paths: dict[str, Path],
        model,
        transform,
        fold_record: dict,
        predictions: pd.DataFrame,
    ):
        paths['checkpoint'].parent.mkdir(parents=True, exist_ok=True)
        prediction_tmp = paths['predictions'].with_name(f"{paths['predictions'].name}.tmp")
        predictions.to_csv(prediction_tmp, index=False, compression='gzip')
        os.replace(prediction_tmp, paths['predictions'])
        self.save_checkpoint(paths['checkpoint'], model, transform)
        # Written last: this record marks a completed fold.
        metrics_tmp = paths['metrics'].with_name(f"{paths['metrics'].name}.tmp")
        with metrics_tmp.open('w', encoding='utf-8') as file:
            json.dump(fold_record, file, indent=2)
        os.replace(metrics_tmp, paths['metrics'])

    def _save_fold_table(self):
        if not self.fold_records:
            return
        table = pd.DataFrame(self.fold_records).sort_values(['task', 'repeat', 'fold'])
        path = self.results_dir / 'fold_metrics.csv'
        temporary_path = path.with_name(f'{path.name}.tmp')
        table.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)

    def get_training_modules(self, dataset: PolymerDataset, seed: int):
        self.seed_everything(seed)
        model = MLP(**self.model_config).to(self.device)
        transform = Transform(dataset.values, dataset.log10_idx, dataset.logm1_idx).to(self.device)

        decayed, no_decay = [], []
        for name, parameter in model.named_parameters():
            (no_decay if name.endswith('bias') else decayed).append(parameter)

        optimizer = torch.optim.AdamW([
            {'params': decayed, 'weight_decay': self.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ], lr=self.learning_rate)
        return model, transform, optimizer

    def get_train_loader(self, dataset: PolymerDataset, seed: int):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=self.drop_last,
        )

    def get_eval_loader(self, dataset: PolymerDataset):
        return DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def print_summary(self):
        records = pd.DataFrame(self.fold_records)
        scores = records.pivot(index=['repeat', 'fold'], columns='task', values='R2')[self.tasks]
        category_means = []
        for props in self.dataset.categories.values():
            props = [prop for prop in props if prop in scores.columns]
            if props:
                category_means.append(scores[props].mean(axis=1))
        if category_means:
            overall = pd.concat(category_means, axis=1).mean(axis=1)
        else:
            overall = scores.values.mean(axis=1)

        ddof = 1 if len(overall) > 1 else 0
        mean, std = np.mean(overall), np.std(overall, ddof=ddof)
        print(f'Average R2: {mean:.3f}±{std:.3f}')

    def _train_kfold(self, task_dir: Path, task: str):
        base_dataset = self.dataset.subset(tasks=[task])
        manifest_path = Path(self.output_dir) / 'splits' / f'{task}.json.gz'
        if manifest_path.is_file():
            manifest = load_manifest(manifest_path)
            validate_manifest(
                manifest,
                base_dataset,
                n_folds=self.n_folds,
                n_trials=self.n_trials,
                seed=self.seed,
                inner_validation_size=self.inner_validation_size,
            )
        else:
            manifest = build_manifest(
                base_dataset,
                n_folds=self.n_folds,
                n_trials=self.n_trials,
                seed=self.seed,
                inner_validation_size=self.inner_validation_size,
            )
            save_manifest(manifest_path, manifest)
        for split in splits_from_manifest(manifest, base_dataset):
            artifact_paths = self._fold_artifact_paths(task_dir, split.repeat, split.fold)
            if self.resume and self._load_completed_fold(artifact_paths):
                print(f'Fold {split.repeat}-{split.fold}: loaded completed result')
                continue

            artifact_paths['metrics'].unlink(missing_ok=True)
            (self.results_dir / 'fold_metrics.csv').unlink(missing_ok=True)
            print(f'Fold {split.repeat}-{split.fold}')
            outer_train_idx = np.union1d(split.inner_train_idx, split.validation_idx)
            train_idx = outer_train_idx if self.validation_only else split.inner_train_idx
            validation_idx = split.test_idx if self.validation_only else split.validation_idx
            train_dataset = base_dataset.subset(indices=train_idx)
            validation_dataset = base_dataset.subset(indices=validation_idx)
            model, transform, optimizer = self.get_training_modules(
                train_dataset,
                split.model_seed,
            )
            train_loader = self.get_train_loader(train_dataset, split.model_seed)
            validation_loader = self.get_eval_loader(validation_dataset)
            last_epoch = (
                ((split.repeat - 1) * self.n_folds + split.fold - 1) * self.max_epochs
            )
            selected_epoch, best_validation_r2 = train(
                model,
                train_loader,
                transform,
                optimizer,
                self.max_epochs,
                last_epoch=last_epoch,
                writer=self.writer,
                eval_loader=validation_loader,
                patience=self.early_stopping_patience,
                min_steps=self.min_steps,
                restore_best=self.validation_only,
            )
            if not self.validation_only:
                del model, transform, optimizer
                outer_train_dataset = base_dataset.subset(indices=outer_train_idx)
                model, transform, optimizer = self.get_training_modules(
                    outer_train_dataset,
                    split.model_seed,
                )
                outer_train_loader = self.get_train_loader(outer_train_dataset, split.model_seed)
                refit_steps = train(
                    model,
                    outer_train_loader,
                    transform,
                    optimizer,
                    selected_epoch,
                    min_steps=self.min_steps,
                )

            evaluation_dataset = base_dataset.subset(indices=split.test_idx)
            evaluation_loader = self.get_eval_loader(evaluation_dataset)
            metrics = evaluate(model, evaluation_loader, transform, pbar=True)

            r2 = metrics['r2'].detach().cpu().numpy()
            rmse = metrics['rmse'].detach().cpu().numpy()
            fold_record = {
                'task': task,
                'repeat': split.repeat,
                'fold': split.fold,
                'selected_epoch': selected_epoch,
                'selected_steps': selected_epoch * len(train_loader),
                'best_validation_r2': best_validation_r2,
                'R2': float(r2[0]),
                'RMSE': float(rmse[0]),
            }
            if self.validation_only:
                fold_record.update(n_train=len(train_dataset), n_validation=len(evaluation_dataset))
            else:
                fold_record.update(
                    refit_steps=refit_steps,
                    n_outer_train=len(outer_train_dataset),
                    n_test=len(evaluation_dataset),
                )
            predictions = pd.DataFrame({
                'task': task,
                'repeat': split.repeat,
                'fold': split.fold,
                'row_id': evaluation_dataset.row_indices,
                'group_id': canonical_group_ids(evaluation_dataset.groups),
                'target': metrics['targets'].detach().cpu().numpy()[:, 0],
                'prediction': metrics['predictions'].detach().cpu().numpy()[:, 0],
            })
            self._save_fold_artifacts(
                artifact_paths,
                model,
                transform,
                fold_record,
                predictions,
            )
            self.fold_records.append(fold_record)
            self._save_fold_table()

    def train(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.results_dir / 'config.json'
        config = self.get_config()
        if self.resume and os.path.isfile(config_path):
            with open(config_path, encoding='utf-8') as f:
                previous = json.load(f)
            if self._resume_settings(previous) != self._resume_settings(config):
                raise ValueError('Training settings changed; use a new output directory or set resume=False')
        if not self.resume:
            (self.results_dir / 'fold_metrics.csv').unlink(missing_ok=True)
            for task in self.tasks:
                for record_path in (self.results_dir / task).glob('fold-*.json'):
                    record_path.unlink()
        temporary_config_path = f'{config_path}.tmp'
        with open(temporary_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        os.replace(temporary_config_path, config_path)

        if self.logging_dir is not None:
            self.writer = SummaryWriter(self.logging_dir)
            self.writer.add_custom_scalars({
                'R2 Score': {
                    category: ['Multiline', [f'R2/{name}' for name in names]]
                    for category, names in self.dataset.categories.items()
                },
            })

        for task in self.tasks:
            print(f'Task: {task}')
            task_dir = self.results_dir / task
            task_dir.mkdir(exist_ok=True)
            self._train_kfold(task_dir, task)
            score = [record['R2'] for record in self.fold_records if record['task'] == task]
            ddof = 1 if len(score) > 1 else 0
            print(f'R2: {np.mean(score):.3f}±{np.std(score, ddof=ddof):.3f}\n')

        self._save_fold_table()
        self.print_summary()
        print(f'Results saved to {self.results_dir / "fold_metrics.csv"}')
        if self.writer is not None:
            self.writer.close()
