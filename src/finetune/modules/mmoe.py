import torch
import torch.nn as nn

from .blocks import Expert


class MMoE(nn.Module):
    def __init__(self, input_dim, num_tasks, expert_dim, num_blocks, num_experts, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_dim, num_blocks, dropout) for _ in range(num_experts)]
        )
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts, bias=False),
                nn.Softmax(dim=-1),
            ) for _ in range(num_tasks)
        ])
        self.need_weights = False
        self.expert_weights = []

    def forward(self, x):
        # [batch_size, num_experts, expert_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        self.expert_weights.clear()
        outputs = []
        for gate in self.gates:
            gate_weights = gate(x)  # [batch_size, num_experts]

            if self.need_weights:
                self.expert_weights.append(gate_weights.detach())
                
            weighted_expert = torch.einsum('be,bed->bd', gate_weights, expert_outputs)  # [batch_size, expert_dim]
            outputs.append(weighted_expert)

        return outputs


class MMoEModel(nn.Module):
    def __init__(self, input_dim, num_tasks, expert_dim=64, tower_dim=32, num_blocks=1, num_experts=4, dropout=0.1):
        super().__init__()
        self.mmoe = MMoE(input_dim, num_tasks, expert_dim, num_blocks, num_experts, dropout)
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, tower_dim),
                nn.LayerNorm(tower_dim),
                nn.GELU(),
                nn.Linear(tower_dim, 1),
            ) for _ in range(num_tasks)
        ])
    
    def forward(self, x):
        mmoe_outputs = self.mmoe(x)  # [task1_feat, task2_feat, ...]
        task_outputs = []
        for features, tower in zip(mmoe_outputs, self.towers):
            task_outputs.append(tower(features))
        
        return torch.cat(task_outputs, dim=1)
