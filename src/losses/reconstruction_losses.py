# src/losses/reconstruction_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.phase_derivative_loss import (
    PhaseDerivativeLoss
)

from src.losses.von_mises_loss import (
    VonMisesLoss
)

from src.losses.phase_continuity_loss import (
    PhaseContinuityLoss
)


class ReconstructionLoss(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.cfg = cfg

        self.current_epoch = 0
        self.total_epochs = 1

        loss_cfg = (
            cfg["losses"]["reconstruction"]
        )

        self.l1_weight = (
            loss_cfg["l1_weight"]
        )

        self.mse_weight = (
            loss_cfg["mse_weight"]
        )

        self.phase_derivative_weight = (
            loss_cfg.get(
                "phase_derivative_weight",
                0.25
            )
        )

        self.von_mises_weight = (
            loss_cfg.get(
                "von_mises_weight",
                0.25
            )
        )

        self.phase_continuity_weight = (
            loss_cfg.get(
                "phase_continuity_weight",
                0.10
            )
        )

        self.phase_derivative_loss = (
            PhaseDerivativeLoss()
        )

        self.von_mises_loss = (
            VonMisesLoss()
        )

        self.phase_continuity_loss = (
            PhaseContinuityLoss()
        )

    def phase_enabled(
        self
    ):

        stage_cfg = (
            self.cfg[
                "staged_activation"
            ]
        )

        if not stage_cfg["enabled"]:

            return True

        progress = (

            self.current_epoch

            /

            max(
                1,
                self.total_epochs
            )
        )

        return (

            progress

            >=

            stage_cfg[
                "phase_start"
            ]
        )

    def align_target(
        self,
        prediction,
        target
    ):

        if target.ndim == 4:

            target = target.squeeze(1)

        if prediction.shape != target.shape:

            target = F.interpolate(

                target,

                size=
                prediction.shape[-1],

                mode="linear",

                align_corners=False
            )

        return target

    def feature_loss(
        self,
        prediction,
        target
    ):

        target = self.align_target(

            prediction,
            target
        )

        l1 = F.l1_loss(
            prediction,
            target
        )

        mse = F.mse_loss(
            prediction,
            target
        )

        total = (

            self.l1_weight * l1 +

            self.mse_weight * mse
        )

        return total, l1, mse

    def forward(
        self,
        reconstructions,
        targets
    ):

        total_loss = 0.0

        metrics = {}

        for feature_name in reconstructions:

            pred = (
                reconstructions[
                    feature_name
                ]
            )

            target = (
                targets[
                    feature_name
                ]
            )

            loss, l1, mse = (

                self.feature_loss(
                    pred,
                    target
                )
            )

            metrics[
                f"{feature_name}_loss"
            ] = loss

            metrics[
                f"{feature_name}_l1"
            ] = l1

            metrics[
                f"{feature_name}_mse"
            ] = mse

            total_loss += loss

        target_sin = self.align_target(

            reconstructions[
                "phase_sin"
            ],

            targets[
                "phase_sin"
            ]
        )

        target_cos = self.align_target(

            reconstructions[
                "phase_cos"
            ],

            targets[
                "phase_cos"
            ]
        )

        if self.phase_enabled():

            derivative_metrics = (

                self.phase_derivative_loss(

                    reconstructions[
                        "phase_sin"
                    ],

                    reconstructions[
                        "phase_cos"
                    ],

                    target_sin,

                    target_cos
                )
            )

            derivative_loss = (

                self.phase_derivative_weight *

                derivative_metrics[
                    "loss"
                ]
            )

            total_loss += derivative_loss

            metrics[
                "phase_derivative"
            ] = derivative_loss

            metrics[
                "derived_if_loss"
            ] = derivative_metrics[
                "if_loss"
            ]

            metrics[
                "derived_gd_loss"
            ] = derivative_metrics[
                "gd_loss"
            ]

            von_mises = (

                self.von_mises_loss(

                    reconstructions[
                        "phase_sin"
                    ],

                    reconstructions[
                        "phase_cos"
                    ],

                    target_sin,

                    target_cos
                )
            )

            von_mises = (

                self.von_mises_weight *

                von_mises
            )

            total_loss += von_mises

            metrics[
                "von_mises"
            ] = von_mises

            continuity = (

                self.phase_continuity_loss(

                    reconstructions[
                        "phase_sin"
                    ],

                    reconstructions[
                        "phase_cos"
                    ],

                    target_sin,

                    target_cos
                )
            )

            continuity = (

                self.phase_continuity_weight *

                continuity
            )

            total_loss += continuity

            metrics[
                "phase_continuity"
            ] = continuity

        else:

            zero = torch.tensor(
                0.0,
                device=target_sin.device
            )

            metrics[
                "phase_derivative"
            ] = zero

            metrics[
                "derived_if_loss"
            ] = zero

            metrics[
                "derived_gd_loss"
            ] = zero

            metrics[
                "von_mises"
            ] = zero

            metrics[
                "phase_continuity"
            ] = zero

        metrics["loss"] = total_loss

        return metrics