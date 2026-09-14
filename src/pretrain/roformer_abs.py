import torch
from torch import nn

from transformers.models.roformer.modeling_roformer import (
    RoFormerModel,
    RoFormerSinusoidalPositionalEmbedding,
)


class SinusoidalAbsoluteEmbeddings(nn.Module):
    def __init__(self, base_embeddings, config):
        super().__init__()

        self.word_embeddings = base_embeddings.word_embeddings
        self.token_type_embeddings = base_embeddings.token_type_embeddings
        self.LayerNorm = base_embeddings.LayerNorm
        self.dropout = base_embeddings.dropout

        # Build on CPU even during from_pretrained's meta-device initialization.
        # Keep RoFormer's [sin..., cos...] layout.
        dim = config.embedding_size
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32, device="cpu")
        frequencies = 10000.0 ** (-torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
        angles = positions[:, None] * frequencies[None, :]
        pe = torch.cat([angles.sin(), angles[:, :dim // 2].cos()], dim=-1)
        self.register_buffer(
            "position_embeddings",
            pe,
            persistent=False,
        )

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=inputs_embeds.device
            )

        embeddings = (
            inputs_embeds
            + self.token_type_embeddings(token_type_ids)
            + self.position_embeddings[:input_shape[1]].to(inputs_embeds.dtype)
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class IdentityRotaryEmbedding(RoFormerSinusoidalPositionalEmbedding):
    @torch.no_grad()
    def forward(self, input_ids_shape, past_key_values_length=0, position_ids=None):
        seq_len = input_ids_shape[1]
        dim = self.embedding_dim

        assert dim % 2 == 0

        sin = self.weight.new_zeros(seq_len, dim // 2)
        cos = self.weight.new_ones(seq_len, dim // 2)

        return torch.cat([sin, cos], dim=-1)


class RoFormerSinusoidalAbsolute(RoFormerModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = SinusoidalAbsoluteEmbeddings(
            self.embeddings, config
        )

        head_dim = config.hidden_size // config.num_attention_heads
        self.encoder.embed_positions = IdentityRotaryEmbedding(
            config.max_position_embeddings,
            head_dim,
        )
