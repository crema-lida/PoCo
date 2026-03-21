import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_hidden_layers=1, dropout=0.1):
        super().__init__()
        self.hidden = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for i in range(num_hidden_layers)
        ])
        self.output = nn.Linear(hidden_dim if num_hidden_layers > 0 else input_dim, 1)
    
    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
        )
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class Expert(nn.Module):
    def __init__(self, input_dim, expert_dim, num_blocks=1, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, expert_dim, bias=False)
        self.net = nn.Sequential(*[
            ResBlock(expert_dim, expert_dim, dropout)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        x = self.projection(x)
        return self.net(x)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.W_q, self.W_k, self.W_v = [
            nn.Linear(embed_dim, embed_dim, bias=bias) for _ in range(3)
        ]
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None
        self.need_weights = False

    def forward(self, target, source):
        batch_size, target_length, _ = target.shape
        source_length = source.shape[1]
        
        Q = self.W_q(target)
        K = self.W_k(source)
        V = self.W_v(source)

        # Q: [B, T, D] -> [B, T, H, D_head] -> [B, H, T, D_head]
        Q = Q.view(batch_size, target_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # K: [B, S, D] -> [B, S, H, D_head] -> [B, H, D_head, S]
        K = K.view(batch_size, source_length, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        
        # V: [B, S, D] -> [B, S, H, D_head] -> [B, H, S, D_head]
        V = V.view(batch_size, source_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # [B, H, T, S]
        attn_scores = torch.matmul(Q, K) / self.head_dim ** 0.5
        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.need_weights:
            self.attn_weights = attn_weights.detach()

        attn_weights = self.dropout(attn_weights)
        
        # [B, H, T, D_head]
        heads_output = torch.matmul(attn_weights, V)

        # Merge heads [B, T, D]
        attn_output = heads_output.permute(0, 2, 1, 3).contiguous().view(batch_size, target_length, -1)

        return self.proj_out(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, ff_factor=4):
        super().__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout, bias)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_factor * embed_dim),
            nn.GELU(),
            nn.Linear(ff_factor * embed_dim, embed_dim)
        )
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm_attn(x)
        attn = self.attention(z, z)
        x = x + self.dropout(attn)
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        return x + self.dropout(ff)
