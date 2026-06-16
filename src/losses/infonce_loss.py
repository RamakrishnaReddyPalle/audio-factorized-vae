# src/losses/infonce_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(
    nn.Module
):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.temperature = (

            cfg["losses"]

            ["contrastive"]

            ["temperature"]
        )

    def forward(
        self,
        z1,
        z2
    ):

        z1 = F.normalize(
            z1,
            dim=-1
        )

        z2 = F.normalize(
            z2,
            dim=-1
        )

        logits = (

            z1 @ z2.T

        ) / self.temperature

        labels = torch.arange(

            z1.shape[0],

            device=z1.device
        )

        return F.cross_entropy(
            logits,
            labels
        )