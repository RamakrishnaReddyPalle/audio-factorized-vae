# src/models/factorized/fidelity_cross_attention.py

import torch
import torch.nn as nn


class FidelityCrossAttention(nn.Module):

    def __init__(
        self,
        hidden_dim,
        fidelity_dim,
        heads=8
    ):

        super().__init__()

        # ----------------------------------
        # Attention memory
        # ----------------------------------

        self.memory_proj = nn.Linear(

            fidelity_dim,

            hidden_dim
        )

        self.attn = nn.MultiheadAttention(

            hidden_dim,

            heads,

            batch_first=True
        )

        # ----------------------------------
        # Fidelity skip token
        #
        # Direct latent → hidden path
        #
        # Prevents fidelity bottleneck
        # through attention only.
        # ----------------------------------

        self.skip_proj = nn.Linear(

            fidelity_dim,

            hidden_dim
        )

        self.skip_gate = nn.Sequential(

            nn.Linear(
                fidelity_dim,
                hidden_dim
            ),

            nn.Sigmoid()
        )

    def forward(

        self,

        x,

        z_fidelity
    ):

        # ----------------------------------
        # Attention branch
        # ----------------------------------

        memory = self.memory_proj(

            z_fidelity
        )

        memory = memory.unsqueeze(1)

        attn_out, _ = self.attn(

            query=x,

            key=memory,

            value=memory
        )

        # ----------------------------------
        # Skip token branch
        #
        # [B,F]
        # →
        # [B,D]
        # →
        # [B,1,D]
        # →
        # [B,T,D]
        # ----------------------------------

        skip_token = self.skip_proj(

            z_fidelity
        )

        gate = self.skip_gate(

            z_fidelity
        )

        skip_token = (

            skip_token
            *
            gate
        )

        skip_token = (

            skip_token
            .unsqueeze(1)
            .expand(
                -1,
                x.shape[1],
                -1
            )
        )

        # ----------------------------------
        # Final fusion
        # ----------------------------------

        x = (

            x
            +
            attn_out
            +
            skip_token
        )

        return x