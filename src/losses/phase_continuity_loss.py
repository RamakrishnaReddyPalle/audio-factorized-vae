# src/losses/phase_continuity_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseContinuityLoss(
    nn.Module
):

    def __init__(self):

        super().__init__()

    def phase_from_sincos(

        self,

        sin_phase,

        cos_phase
    ):

        return torch.atan2(

            sin_phase,

            cos_phase
        )

    def temporal_derivative(

        self,

        phase
    ):

        return (

            phase[..., 1:]

            -

            phase[..., :-1]
        )

    def forward(

        self,

        pred_sin,

        pred_cos,

        target_sin,

        target_cos
    ):

        pred_phase = (

            self.phase_from_sincos(

                pred_sin,

                pred_cos
            )
        )

        target_phase = (

            self.phase_from_sincos(

                target_sin,

                target_cos
            )
        )

        pred_delta = (

            self.temporal_derivative(

                pred_phase
            )
        )

        target_delta = (

            self.temporal_derivative(

                target_phase
            )
        )

        wrapped_error = (

            1.0 -

            torch.cos(

                pred_delta -

                target_delta
            )
        )

        loss = wrapped_error.mean()

        return loss