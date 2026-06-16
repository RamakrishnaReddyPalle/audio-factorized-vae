# src/models/factorized/cross_scale_attention.py

import torch
import torch.nn as nn


class CrossScaleAttention(nn.Module):

    def __init__(
        self,
        dim,
        heads
    ):

        super().__init__()

        self.attn = nn.MultiheadAttention(

            embed_dim=dim,

            num_heads=heads,

            batch_first=True
        )

        self.norm = nn.LayerNorm(
            dim
        )

    def forward(
        self,
        x
    ):

        attn_out, _ = self.attn(

            x,
            x,
            x
        )

        x = self.norm(
            x + attn_out
        )

        return x