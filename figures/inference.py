import os
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

import sys
sys.path.append('../src/')

from finetune.modules import MLP, Transform


def mean_pooling(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()  # (B, S, 1)
    return (hidden * mask).sum(1) / mask.sum(1)


class EnsembleModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        task_heads: nn.ModuleList,
        transforms: nn.ModuleList,
    ):
        super().__init__()
        self.encoder = encoder
        self.ensemble = task_heads
        self.transforms = transforms

    def forward(self, input_ids, attention_mask, inverse=False, return_std=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeds = mean_pooling(outputs.last_hidden_state, attention_mask)
        predictions = []
        for (model, transform) in zip(self.ensemble, self.transforms):
            pred = model(embeds)
            if inverse:
                pred = transform(pred.to(torch.float64), inverse=inverse)
            predictions.append(pred)
        predictions = torch.stack(predictions, dim=-1)
        pred = predictions.mean(dim=-1)
        if return_std:
            return pred, predictions.std(dim=-1, unbiased=False)
        return pred


def get_ensemble_model(checkpoint_path: str, task_name: str) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)

    with open(checkpoint_path / 'config.json') as f:
        config = json.load(f)

    encoder_path = '..' / Path(config['dataset']['encoder']['path'])
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    encoder = AutoModel.from_pretrained(encoder_path)

    model_config: dict = config['model_config']
    ensemble = nn.ModuleList()
    transforms = nn.ModuleList()
    for ckpt_file in os.listdir(checkpoint_path / task_name):
        state = torch.load(checkpoint_path / task_name / ckpt_file)
        model = MLP(**model_config)
        model.load_state_dict(state['model'])
        transform = Transform()
        transform.load_state_dict(state['transform'])
        ensemble.append(model)
        transforms.append(transform)

    model = EnsembleModel(encoder, ensemble, transforms).eval()
    return model, tokenizer
