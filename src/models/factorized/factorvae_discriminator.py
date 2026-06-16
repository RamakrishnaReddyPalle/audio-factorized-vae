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

        return self.net(
            z
        )