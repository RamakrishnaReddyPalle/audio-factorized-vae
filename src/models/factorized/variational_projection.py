# src/models/factorized/variational_projection.py

import torch
import torch.nn as nn


# --------------------------------------------------
# Standard Variational Head
# --------------------------------------------------

class VariationalProjection(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim
    ):

        super().__init__()

        self.pre_projection = nn.Sequential(

            nn.Linear(
                input_dim,
                input_dim
            ),

            nn.LayerNorm(
                input_dim
            ),

            nn.GELU()
        )

        self.mu = nn.Linear(

            input_dim,

            latent_dim
        )

        self.logvar = nn.Linear(

            input_dim,

            latent_dim
        )

    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(
            std
        )

        return mu + eps * std

    def forward(
        self,
        x
    ):

        x = self.pre_projection(
            x
        )

        mu = self.mu(
            x
        )

        logvar = self.logvar(
            x
        )

        z = self.reparameterize(

            mu,

            logvar
        )

        return {

            "z":
                z,

            "mu":
                mu,

            "logvar":
                logvar
        }


# --------------------------------------------------
# Deterministic Projection
# --------------------------------------------------

class DeterministicProjection(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                latent_dim
            ),

            nn.LayerNorm(
                latent_dim
            ),

            nn.GELU(),

            nn.Linear(
                latent_dim,
                latent_dim
            )
        )

    def forward(
        self,
        x
    ):

        z = self.net(
            x
        )

        return {

            "z": z
        }