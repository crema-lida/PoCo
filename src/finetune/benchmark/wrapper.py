from pathlib import Path
from typing import Callable
from numpy.typing import NDArray
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, RobertaModel

BENCH_DIR = Path(__file__).resolve().parent


def mean_pooling(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()  # (B, S, 1)
    return (hidden * mask).sum(1) / mask.sum(1)


def cls_pooling(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return hidden[:, 0, :]


class PolyCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_config(config)

    def forward(self, **kwargs):
        return self.encoder(**kwargs)


def get_encoder(encoder_path: str) -> Callable[[list[str]], NDArray]:
    bench_path = BENCH_DIR / encoder_path

    match encoder_path:
        case 'polyBERT':
            model = AutoModel.from_pretrained(bench_path, output_hidden_states=True)
            tokenizer = AutoTokenizer.from_pretrained(bench_path)
            max_length = model.config.max_position_embeddings

        case 'TransPolymer':
            from .TransPolymer.PolymerSmilesTokenization import PolymerSmilesTokenizer

            model = RobertaModel.from_pretrained(bench_path, output_hidden_states=True)
            tokenizer = PolymerSmilesTokenizer.from_pretrained('roberta-base', local_files_only=True)
            max_length = model.config.max_position_embeddings

        case 'PolyCL':
            model_config = AutoConfig.from_pretrained(bench_path)
            model_config.output_hidden_states = True
            model = PolyCL(model_config)
            model.load_state_dict(torch.load(bench_path / 'polycl.pth'), strict=False)
            tokenizer = AutoTokenizer.from_pretrained(bench_path)
            max_length = model.encoder.config.max_position_embeddings
            
        case 'MMPolymer':
            from .MMPolymer.encoder import build_mm_polymer_encoder
            return build_mm_polymer_encoder(bench_path)

        case 'PerioGT':
            from .PerioGT.encoder import build_periogt_encoder
            import os
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['OMP_NUM_THREADS'] = '1'
            return build_periogt_encoder(bench_path)

        case _:
            config = AutoConfig.from_pretrained(encoder_path, output_hidden_states=True)
            model_cls = AutoModel
            if config.model_type == 'roformer' and getattr(config, 'position_embedding_type', 'rotary') == 'absolute':
                import sys
                sys.path.append(str(BENCH_DIR.parents[1]))
                from pretrain.roformer_abs import RoFormerSinusoidalAbsolute

                model_cls = RoFormerSinusoidalAbsolute
            model = model_cls.from_pretrained(encoder_path, config=config)
            tokenizer = AutoTokenizer.from_pretrained(encoder_path)
            max_length = model.config.max_position_embeddings
    
    model.eval().cuda()
    
    @torch.no_grad
    def encode(
        sentences: list[str],
        pooling='mean',
        batch_size=64,
        concat_last_layers: int | None = None,
        **kwargs,
    ) -> NDArray:
        match pooling:
            case 'mean':
                pool_func = mean_pooling
            case 'cls':
                pool_func = cls_pooling
            case _:
                raise ValueError(f'Unknown pooling method: {pooling}')
        
        outputs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            out = model(**enc.to('cuda'))

            if concat_last_layers is not None:
                hidden_states = out.hidden_states[-concat_last_layers:]
                hidden = torch.concatenate(hidden_states, dim=-1)  # (B, S, D * L)
                embeds = pool_func(hidden, enc['attention_mask'])  # (B, D * L)
            else:
                embeds = pool_func(out.last_hidden_state, enc['attention_mask'])  # (B, D)
            
            outputs.append(embeds)
        return torch.cat(outputs).cpu().numpy()

    return encode
