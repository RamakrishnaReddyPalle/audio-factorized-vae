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

        zf = latents["fidelity"]

        pair_losses = {}

        pair_losses["content_speaker"] = (
            self.pair_loss(
                zc,
                zs
            )
        )

        pair_losses["content_environment"] = (
            self.pair_loss(
                zc,
                ze
            )
        )

        pair_losses["content_excitation"] = (
            self.pair_loss(
                zc,
                zx
            )
        )

        pair_losses["content_fidelity"] = (
            self.pair_loss(
                zc,
                zf
            )
        )

        pair_losses["speaker_environment"] = (
            self.pair_loss(
                zs,
                ze
            )
        )

        pair_losses["speaker_excitation"] = (
            self.pair_loss(
                zs,
                zx
            )
        )

        pair_losses["speaker_fidelity"] = (
            self.pair_loss(
                zs,
                zf
            )
        )

        pair_losses["environment_excitation"] = (
            self.pair_loss(
                ze,
                zx
            )
        )

        pair_losses["environment_fidelity"] = (
            self.pair_loss(
                ze,
                zf
            )
        )

        pair_losses["excitation_fidelity"] = (
            self.pair_loss(
                zx,
                zf
            )
        )

        total_loss = sum(
            pair_losses.values()
        )

        return total_loss