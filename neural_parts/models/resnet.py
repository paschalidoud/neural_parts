import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_dims, out_dims, norm_method="batch_norm"):
        super().__init__()
        self.proj = (
            nn.Sequential() if in_dims == out_dims else
            nn.Linear(in_dims, out_dims)
        )
        self.fc1 = nn.Linear(out_dims, out_dims)
        self.fc2 = nn.Linear(out_dims, out_dims)

        if norm_method == "layer_norm":
            self.norm1 = nn.LayerNorm(out_dims)
            self.norm2 = nn.LayerNorm(out_dims)
        elif norm_method == "no_norm":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
        else:
            raise ValueError(("Invalid normalization method "
                              "{}").format(norm_method))

    def _norm(self, norm_layer, x):
        B, N, M, D = x.shape
        x = x.permute(0, 3, 1, 2)
        return norm_layer(x).permute(0, 2, 3, 1).contiguous()

    def forward(self, x):
        x = self.proj(x)
        out = self.fc1(x)
        out = self.norm1(out).relu()
        out = self.fc2(out)
        out = self.norm2(out).relu()
        return out + x
