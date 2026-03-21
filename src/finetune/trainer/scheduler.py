from copy import deepcopy
import torch
from torch.optim import Optimizer


class PlateauRewindLR:
    """
    Reduce learning rate and restore the best checkpoint (model + optimizer)
    when a metric has stopped improving.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        max_reductions: int = 5,
        min_lr: float = 0,
    ):
        assert mode in ('min', 'max')
        assert factor > 0 and factor < 1.0

        self.model = model
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.max_reductions = max_reductions
        self.min_lr = min_lr

        self.steps_no_improve = 0
        self.num_reductions = 0
        self._last_lr = self.get_last_lr()
        self.early_stop = False

        self.best_metric = torch.inf if mode == 'min' else -torch.inf
        self.best_state = self._get_state()

    def step(self, metric: float) -> list[float]:
        self._update_best_state(metric)
        if self.steps_no_improve < self.patience:
            return self._last_lr
        
        self._last_lr = self.get_last_lr()
        can_reduce = (self.num_reductions < self.max_reductions) and any(
            lr > self.min_lr for lr in self._last_lr
        )
        if not can_reduce:
            self.early_stop = True
            self.load_best_state()
            return self._last_lr
        
        new_lr = [max(lr * self.factor, self.min_lr) for lr in self._last_lr]
        self.load_best_state()
        for group, lr in zip(self.optimizer.param_groups, new_lr):
            group['lr'] = lr

        self.num_reductions += 1
        self.steps_no_improve = 0
        return new_lr

    def get_last_lr(self) -> list[float]:
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def load_best_state(self):
        self._load_state(self.best_state)

    def _update_best_state(self, metric: float) -> bool:
        if self.mode == 'min':
            improved = metric < self.best_metric * (1 - self.threshold)
        else:
            improved = metric > self.best_metric * (1 + self.threshold)

        if improved:
            self.best_metric = metric
            self.best_state = self._get_state()
            self.steps_no_improve = 0
        else:
            self.steps_no_improve += 1

    def _get_state(self) -> dict[str]:
        return {
            'model': deepcopy(self.model.state_dict()),
            'optimizer': deepcopy(self.optimizer.state_dict()),
        }

    def _load_state(self, state: dict[str]):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
