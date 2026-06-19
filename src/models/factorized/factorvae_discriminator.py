# src/models/factorized/factorvae_discriminator.py

import torch
import torch.nn as nn


class FactorVAEDiscriminator(
    nn.Module
):

    def __init__(
        self,
        latent_dim,
        hidden_dim=1000
    ):

        super().__init__()

        self.latent_dim = (
            latent_dim
        )

        self.hidden_dim = (
            hidden_dim
        )

        self.net = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Linear(
                hidden_dim,
                2
            )
        )

    def forward(
        self,
        z
    ):

        if z.ndim != 2:

            raise ValueError(

                f"FactorVAEDiscriminator "
                f"expected [B,D], got "
                f"{tuple(z.shape)}"
            )

        if z.shape[-1] != self.latent_dim:

            raise ValueError(

                f"Expected latent dim "
                f"{self.latent_dim}, got "
                f"{z.shape[-1]}"
            )

        logits = self.net(
            z
        )

        return logits