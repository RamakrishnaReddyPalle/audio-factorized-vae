# src/losses/von_mises_loss.py

import torch
import torch.nn as nn


class VonMisesLoss(nn.Module):

    def __init__(
        self,
        kappa=1.0
    ):

        super().__init__()

        self.kappa = kappa

    def forward(

        self,

        pred_sin,
        pred_cos,

        target_sin,
        target_cos
    ):

        pred_phase = torch.atan2(

            pred_sin,
            pred_cos
        )

        target_phase = torch.atan2(

            target_sin,
            target_cos
        )

        delta = (
            pred_phase -
            target_phase
        )

        loss = (

            1.0 -

            torch.cos(
                delta
            )
        )

        return loss.mean()