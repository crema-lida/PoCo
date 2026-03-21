from dataclasses import dataclass, asdict
import os
import json
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from modules import MultiTaskGroup, MLP, MultiTaskModel, MMoEModel, Transform
from .dataset import MTL, RadonPy, PolyOmics
from .train import train
from .evaluate import evaluate
from .scheduler import PlateauRewindLR

PolymerDataset = MTL | RadonPy | PolyOmics


@dataclass
class MultiTaskTrainer:
    model_config: dict | None = None
    dataset_config: dict | None = None
    encoder_path: str | None = None
    pooling: str = 'mean'
    concat_last_layers: int | None = None
    serial: bool = True
    output_dir: str = './checkpoints'
    logging_dir: str | None = None
    pretrained_dir: str | None = None
    freeze_shared_layers: bool = False
    device: str = 'cuda'
    n_folds: int = 1
    n_trials: int = 1
    seed: int = 42
    max_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0
    train_batch_size: int = 64
    eval_batch_size: int = 1024
    eval_size: int | float = 0.1
    num_workers: int = 0
    drop_last: bool = False

    def __post_init__(self):
        if self.pretrained_dir is not None:
            with open(os.path.join(self.pretrained_dir, 'config.json')) as f:
                pretrained_config = json.load(f)
                self.model_config = pretrained_config['model_config']

        dataset_name = self.dataset_config['name']
        self.dataset: PolymerDataset = eval(dataset_name)
        assert issubclass(self.dataset, PolymerDataset), f'Unknown dataset: {dataset_name}'
        self.dataset.encode_smiles(
            self.dataset_config['path'],
            self.encoder_path,
            pooling=self.pooling,
            concat_last_layers=self.concat_last_layers,
        )
        self.tasks = self.dataset_config.get('tasks', self.dataset.properties)
        self._train = self._train_kfold if self.n_folds > 1 else self._train_full
        
        self.writer: SummaryWriter | None = None
        self.results = {metric: {prop: [] for prop in self.tasks} for metric in ['R2', 'MAE']}

    def save_checkpoint(self, filepath, model, transform):
        torch.save({
            'model': model.state_dict(),
            'transform': transform.state_dict(),
        }, filepath)

    def get_pretrained_model(self, path):
        checkpoint = torch.load(path)
        model = self.get_model(checkpoint['model'])
        transform = Transform(num_features=len(self.dataset.properties)).to(self.device)
        transform.load_state_dict(checkpoint['transform'])
        return model, transform

    def get_model(self, state_dict=None):
        config = self.model_config.copy()
        model_type = config.pop('model_type', 'ST')

        match model_type:
            case 'ST':
                if len(self.dataset.properties) > 1:
                    model = MultiTaskGroup()
                    for _ in self.dataset.properties:
                        model.models.append(MLP(**config))
                else:
                    model = MLP(**config)
            case 'MT':
                model = MultiTaskModel(num_tasks=len(self.dataset.properties), **config)
                if self.freeze_shared_layers:
                    model.shared.requires_grad_(False)
            case 'MMoE':
                model = MMoEModel(num_tasks=len(self.dataset.properties), **config)
                if self.freeze_shared_layers:
                    model.mmoe.experts.requires_grad_(False)
            case _:
                raise ValueError(f'Unknown model: {model_type}')
        
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
        
        return model.to(self.device)
    
    def get_training_modules(self, data_loader):
        if self.pretrained_dir:
            checkpoint = torch.load(f'{self.pretrained_dir}/full.pth')
            state_dict = {
                k: v for k, v in checkpoint['model'].items()
                if 'experts' in k or 'shared' in k
            }
        else:
            state_dict = None
            
        model = self.get_model(state_dict)
        dataset = data_loader.dataset
        transform = Transform(dataset.values, dataset.log10_idx, dataset.logm1_idx).to(self.device)

        decayed, no_decay = [], []
        for n, p in model.named_parameters():
            (no_decay if n.endswith('bias') else decayed).append(p)
        
        optimizer = torch.optim.AdamW([
            {'params': decayed, 'weight_decay': self.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=self.learning_rate)
        scheduler = PlateauRewindLR(model, optimizer, mode='max', patience=50, max_reductions=0)
        return model, transform, optimizer, scheduler
    
    def get_train_loader(self, indices=None):
        return DataLoader(
            self.dataset(indices),
            batch_size=self.train_batch_size,
            shuffle=True,
            generator=torch.manual_seed(self.seed),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=self.drop_last,
        )
    
    def get_eval_loader(self, indices=None):
        return DataLoader(
            self.dataset(indices),
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
    
    def add_results(self, r2, mae):
        for i, task in enumerate(self.dataset.properties):
            self.results['R2'][task].append(r2[i])
            self.results['MAE'][task].append(mae[i])

    def save_results(self, path):
        df1 = pd.DataFrame(self.results['R2'])
        df2 = pd.DataFrame(self.results['MAE'])
        category_means = []
        for props in self.dataset.categories.values():
            props = [p for p in props if p in df1.columns]
            if props:
                category_means.append(df1[props].mean(axis=1))
        if category_means:
            overall = pd.concat(category_means, axis=1).mean(axis=1)
        else:
            overall = df1.values.mean(axis=1)
        df1['Overall'] = overall
        df1.set_index('Overall', inplace=True)
        with pd.ExcelWriter(path) as writer:
            df1.to_excel(writer, sheet_name='R2')  
            df2.to_excel(writer, sheet_name='MAE', index=False)

        ddof = 1 if len(overall) > 1 else 0
        mean, std = np.mean(overall), np.std(overall, ddof=ddof)
        print(f'Average R2: {mean:.3f}±{std:.3f}\n'
              f'Results saved to {path}')

    def _train_kfold(self, output_dir):
        random.seed(self.seed)
        seeds = random.sample(range(1000), self.n_trials)
        train_indices = list(range(len(self.dataset())))

        for trial, seed in enumerate(seeds, start=1):
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_indices), start=1):
                train_loader = self.get_train_loader(train_idx)
                eval_loader = self.get_eval_loader(val_idx)
                model, transform, optimizer, scheduler = self.get_training_modules(train_loader)

                last_epoch = (fold - 1) * self.max_epochs
                print(f'Fold {trial}-{fold}')

                train(model, train_loader, transform, optimizer, scheduler, self.max_epochs,
                      last_epoch, self.writer, eval_loader)
                _, r2, mae = evaluate(model, eval_loader, transform, pbar=True, mae=True)
                
                self.add_results(r2.cpu().numpy(), mae.cpu().numpy())
                self.save_checkpoint(f'{output_dir}/fold-{trial}.{fold}.pth', model, transform)
    
    def _train_full(self, output_dir):
        indices = list(range(len(self.dataset())))
        train_idx, val_idx = train_test_split(indices, test_size=self.eval_size, random_state=self.seed)
        train_loader = self.get_train_loader(train_idx)
        eval_loader = self.get_eval_loader(val_idx)
        model, transform, optimizer, scheduler = self.get_training_modules(train_loader)
        
        train(model, train_loader, transform, optimizer, scheduler, self.max_epochs,
              writer=self.writer, eval_loader=eval_loader)
        self.save_checkpoint(f'{output_dir}/full.pth', model, transform)

        _, r2, mae = evaluate(model, eval_loader, transform, pbar=True, mae=True)
        self.add_results(r2.cpu().numpy(), mae.cpu().numpy())

    def _train(self, output_dir): pass

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

        if self.logging_dir is not None:
            self.writer = SummaryWriter(self.logging_dir)
            self.writer.add_custom_scalars({
                'R2 Score': {category: ['Multiline', [f'R2/{name}' for name in names]]
                             for category, names in self.dataset.categories.items()},
            })
        
        if self.serial:
            for task in self.tasks:
                print(f'Task: {task}')
                self.dataset.properties = [task]
                os.makedirs(output_dir := f'{self.output_dir}/{task}', exist_ok=True)
                self._train(output_dir)
                
                score = self.results['R2'][task]
                ddof = 1 if len(score) > 1 else 0
                mean, std = np.mean(score), np.std(score, ddof=ddof)
                print(f'R2: {mean:.3f}±{std:.3f}\n')
            self.dataset.properties = self.tasks
        else:
            self._train(self.output_dir)
        
        self.save_results(f'{self.output_dir}/eval.xlsx')
        if self.writer is not None:
            self.writer.close()
        print('Finished Training.\n')
