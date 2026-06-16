# src/models/factorized/environment_film.py

import torch
import torch.nn as nn


class EnvironmentFiLM(nn.Module):

    def __init__(
        self,
        env_dim,
        hidden_dim
    ):

        super().__init__()

        self.gamma = nn.Linear(
            env_dim,
            hidden_dim
        )

        self.beta = nn.Linear(
            env_dim,
            hidden_dim
        )

    def forward(

        self,

        x,

        z_environment
    ):

        gamma = self.gamma(
            z_environment
        )

        beta = self.beta(
            z_environment
        )

        gamma = gamma.unsqueeze(1)

        beta = beta.unsqueeze(1)

        return gamma * x + beta