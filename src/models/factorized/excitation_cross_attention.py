# src/models/factorized/excitation_cross_attention.py

import torch
import torch.nn as nn


class ExcitationCrossAttention(nn.Module):

    def __init__(
        self,
        hidden_dim,
        excitation_dim,
        heads=8
    ):

        super().__init__()

        self.proj = nn.Linear(

            excitation_dim,

            hidden_dim
        )

        self.attn = nn.MultiheadAttention(

            hidden_dim,

            heads,

            batch_first=True
        )

    def forward(

        self,

        x,

        z_excitation
    ):

        memory = self.proj(
            z_excitation
        )

        memory = memory.unsqueeze(1)

        out, _ = self.attn(

            query=x,

            key=memory,

            value=memory
        )

        return x + out