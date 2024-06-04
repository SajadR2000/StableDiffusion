import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: (B, S, C)
        input_shape = x.shape
        batch_size, seq_len, embed_dim = input_shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        # (B, S, C) -> (B, S, 3 * C) -> 3 * (B, S, C)
        query, key, value = self.in_proj(x).chunk(3, dim=-1)
        # (B, S, C) -> (B, H, S, Dim/H)
        query = query.view(interim_shape).transpose(1, 2)
        key = key.view(interim_shape).transpose(1, 2)
        value = value.view(interim_shape).transpose(1, 2)
        # (B, H, S, Dim/H) @ (B, H, Dim/H, S) -> (B, H, S, S
        weight = query @ key.transpose(-1, -2)

        if causal_mask:
            # (B, H, S, S) -> (B, H, S, S)
            mask = torch.ones_like(weight).triu_(diagonal=1)
            weight = weight.masked_fill(mask != 0, float('-inf'))

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        # (B, H, S, S) @ (B, H, S, Dim/H) -> (B, H, S, Dim/H)
        output = weight @ value
        # (B, H, S, Dim/H) -> (B, S, H, Dim/H) -> (B, S, C)
        output = output.transpose(1, 2).reshape(input_shape)
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # x: (B, S_q, C_q)
        # y: (B, S_len_kv, C_kv)

        input_shape = x.shape
        batch_size, seq_len, embed_dim = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).reshape(input_shape)
        output = self.out_proj(output)

        return output