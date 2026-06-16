# src/losses/factorvae_loss.py

import torch
import torch.nn as nn


class FactorVAELoss(
    nn.Module
):

    def __init__(
        self,
        gamma
    ):

        super().__init__()

        self.gamma = gamma

    def forward(
        self,
        discriminator_logits
    ):

        tc_estimate = (

            discriminator_logits[:, 0]

            -

            discriminator_logits[:, 1]

        ).mean()

        return (

            self.gamma

            *

            tc_estimate
        )