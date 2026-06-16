# src/losses/environment_consistency_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvironmentConsistencyLoss(
    nn.Module
):

    def __init__(
        self
    ):
        super().__init__()

    def forward(
        self,
        source_environment_latent,
        reconstructed_environment_latent
    ):

        return F.mse_loss(

            source_environment_latent,

            reconstructed_environment_latent
        )