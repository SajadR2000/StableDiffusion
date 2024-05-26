import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # (B, Seq_len) -> (B, Seq_len, Dim)
        x = self.token_embedding(tokens)
        x += self.positional_embedding
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, d_embed: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(d_embed)
        self.attention = SelfAttention(n_heads, d_embed)
        self.layernorm_2 = nn.LayerNorm(d_embed)
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, Seq_len, Dim)
        residual = x 

        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residual

        residual = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        # x = F.gelu(x)
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear_2(x)
        x += residual
        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.type(torch.long)

        # (B, Seq_len) -> (B, Seq_len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        # (B, Seq_len, Dim) -> (B, Seq_len, Dim)
        state = self.layernorm(state)

        return state