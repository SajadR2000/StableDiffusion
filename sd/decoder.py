import torch
from torch import nn
from torch.nn import functional as F
from attention impot SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.attention =  SelfAttention(1, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        residue = x
        # x: (B, C, H, W)
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        # (B, C, H, W) -> (B, C, H*W)
        x = x.view(n, c, h * w)
        # (B, C, H*W) -> (B, H*W, C)
        x = x.transpose(-1, -2)
        # (B, H*W, C) -> (B, H*W, C)
        x = self.attention(x)
        # (B, H*W, C) -> (B, C, H*W)
        x = x.transpose(-1, -2)
        # (B, C, H*W) -> (B, C, H, W)
        x = x.view(n, c, h, w)
        # (B, C, H, W) -> (B, C, H, W)
        x += residue
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        # self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        # self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        # self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        residue = x
        # x = self.groupnorm_1(x)
        # x = F.silu(x)
        # x = self.conv_1(x)
        # x = self.groupnorm_2(x)
        # x = F.silu(x)
        # x = self.conv_2(x)
        x = self.seq(x)
        x = x + self.residual_layer(residue)
        return x