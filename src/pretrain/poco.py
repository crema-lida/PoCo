import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RoFormerPreTrainedModel, RoFormerModel


def mean_pooling(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # hidden: (B, L, D); mask: (B, L)
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_hidden_layers: int):
        super().__init__()
        self.hidden = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.GELU(),
            ) for i in range(num_hidden_layers)
        ])
        self.output = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(x))


class PolymerContrastModel(RoFormerPreTrainedModel):
    def __init__(self, config):
        config.update({'proj_dim': getattr(config, 'proj_dim', 256)})
        config.update({'proj_hidden_layers': getattr(config, 'proj_hidden_layers', 1)})
        super().__init__(config)
        self.proj_dim = config.proj_dim
        self.roformer = RoFormerModel(config)
        self.proj = MLP(config.hidden_size, config.intermediate_size, config.proj_dim, config.proj_hidden_layers)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> dict:
        encoder_outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = encoder_outputs.last_hidden_state  # (B, L, D)
        h = mean_pooling(hidden, attention_mask)  # (B, D)
        z = self.proj(h)
        z = F.normalize(z)
        return {'z': z, 'hidden': hidden}
