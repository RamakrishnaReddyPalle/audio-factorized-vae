# src/losses/phase_derivative_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseDerivativeLoss(nn.Module):

    def __init__(
        self,
        eps=1e-8
    ):

        super().__init__()

        self.eps = eps

    # ----------------------------------
    # Phase Reconstruction
    # ----------------------------------

    def phase_from_sincos(

        self,

        phase_sin,

        phase_cos
    ):

        return torch.atan2(

            phase_sin,

            phase_cos
        )

    # ----------------------------------
    # Circular Difference
    # ----------------------------------

    def wrapped_difference(

        self,

        x
    ):

        return torch.atan2(

            torch.sin(x),

            torch.cos(x)
        )

    # ----------------------------------
    # IF
    # ----------------------------------

    def instantaneous_frequency(

        self,

        phase
    ):

        diff = (

            phase[..., 1:]

            -

            phase[..., :-1]
        )

        diff = self.wrapped_difference(
            diff
        )

        diff = F.pad(

            diff,

            (1, 0)
        )

        return diff

    # ----------------------------------
    # GD
    # ----------------------------------

    def group_delay(

        self,

        phase
    ):

        diff = (

            phase[:, 1:, :]

            -

            phase[:, :-1, :]
        )

        diff = self.wrapped_difference(
            diff
        )

        diff = F.pad(

            diff,

            (0, 0, 1, 0)
        )

        return diff

    # ----------------------------------
    # Stable Normalization
    #
    # Solves IF explosion risk.
    # ----------------------------------

    def normalize(

        self,

        x
    ):

        mean = (

            x.mean(

                dim=(-2, -1),

                keepdim=True
            )
        )

        std = (

            x.std(

                dim=(-2, -1),

                keepdim=True
            )

            + self.eps
        )

        return (

            x - mean
        ) / std

    # ----------------------------------
    # Loss
    # ----------------------------------

    def normalized_l1(

        self,

        pred,

        target
    ):

        pred = self.normalize(
            pred
        )

        target = self.normalize(
            target
        )

        return F.l1_loss(

            pred,

            target
        )

    # ----------------------------------
    # Forward
    # ----------------------------------

    def forward(

        self,

        pred_phase_sin,

        pred_phase_cos,

        target_phase_sin,

        target_phase_cos
    ):

        pred_phase = (

            self.phase_from_sincos(

                pred_phase_sin,

                pred_phase_cos
            )
        )

        target_phase = (

            self.phase_from_sincos(

                target_phase_sin,

                target_phase_cos
            )
        )

        # ----------------------------------
        # Derived IF
        # ----------------------------------

        pred_if = (

            self.instantaneous_frequency(
                pred_phase
            )
        )

        target_if = (

            self.instantaneous_frequency(
                target_phase
            )
        )

        # ----------------------------------
        # Derived GD
        # ----------------------------------

        pred_gd = (

            self.group_delay(
                pred_phase
            )
        )

        target_gd = (

            self.group_delay(
                target_phase
            )
        )

        # ----------------------------------
        # Normalized losses
        # ----------------------------------

        if_loss = self.normalized_l1(

            pred_if,

            target_if
        )

        gd_loss = self.normalized_l1(

            pred_gd,

            target_gd
        )

        total = (

            if_loss

            +

            gd_loss
        )

        return {

            "loss":
                total,

            "if_loss":
                if_loss,

            "gd_loss":
                gd_loss
        }