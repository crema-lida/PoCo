import torch
import torch.nn as nn

from .blocks import Expert


class MultiTaskGroup(nn.Module):
    def __init__(self, modules: list[nn.Module] | None = None):
        super().__init__()
        self.models = nn.ModuleList(modules)

    def forward(self, x):
        return torch.cat([model(x) for model in self.models], dim=1)


class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, num_tasks, shared_dim=256, tower_dim=128, num_blocks=1):
        super().__init__()
        self.shared = Expert(input_dim, shared_dim, num_blocks)
        self.towers = nn.ModuleList()
        for _ in range(num_tasks):
            self.towers.append(nn.Sequential(
                nn.LayerNorm(shared_dim),
                nn.GELU(),
                nn.Linear(shared_dim, tower_dim),
                nn.LayerNorm(tower_dim),
                nn.GELU(),
                nn.Linear(tower_dim, 1),
            ))

    def forward(self, x):
        x = self.shared(x)
        return torch.cat([head(x) for head in self.towers], dim=1)


class MultiTaskLoss(nn.Module):
    """
    Uncertainty weighting for multi-task learning. 
    See https://doi.org/10.48550/arXiv.1705.07115
    """

    def __init__(self, num_tasks=None, log_vars=None):
        super().__init__()
        log_vars = torch.zeros(num_tasks) if log_vars is None else torch.as_tensor(log_vars)
        self.log_vars = nn.Parameter(log_vars)

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            weight = torch.exp(-self.log_vars[i])
            total_loss += loss * weight + self.log_vars[i]
        return total_loss * 0.5
