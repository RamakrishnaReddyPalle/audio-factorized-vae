# src/losses/orthogonality_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalityLoss(
    nn.Module
):

    def __init__(
        self
    ):
        super().__init__()

    def pair_loss(
        self,
        z1,
        z2
    ):

        # ----------------------------------
        # project to common size
        # ----------------------------------

        common_dim = min(

            z1.shape[-1],

            z2.shape[-1]
        )

        z1 = z1[..., :common_dim]

        z2 = z2[..., :common_dim]

        z1 = F.normalize(
            z1,
            dim=-1
        )

        z2 = F.normalize(
            z2,
            dim=-1
        )

        similarity = (

            z1 * z2

        ).sum(
            dim=-1
        )

        return (
            similarity.pow(2)
        ).mean()

    def forward(
        self,
        latents
    ):

        zc = latents["content"]

        zs = latents["speaker"]

        ze = latents["environment"]

        zx = latents["excitation"]

        loss = 0.0

        loss += self.pair_loss(
            zc,
            zs
        )

        loss += self.pair_loss(
            zc,
            ze
        )

        loss += self.pair_loss(
            zs,
            ze
        )

        loss += self.pair_loss(
            zs,
            zx
        )

        loss += self.pair_loss(
            ze,
            zx
        )

        return loss