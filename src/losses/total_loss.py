# src/losses/total_loss.py

import torch
import torch.nn as nn

from src.losses.reconstruction_losses import (
    ReconstructionLoss
)

from src.losses.multires_stft_loss import (
    MultiResolutionFeatureLoss
)

from src.losses.orthogonality_loss import (
    OrthogonalityLoss
)

from src.losses.factorvae_loss import (
    FactorVAELoss
)


class TotalLoss(
    nn.Module
):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.cfg = cfg

        self.current_epoch = 0
        self.total_epochs = 1

        self.reconstruction = (
            ReconstructionLoss(cfg)
        )

        self.multires = (
            MultiResolutionFeatureLoss()
        )

        self.orthogonality = (
            OrthogonalityLoss()
        )

        self.factorvae = (
            FactorVAELoss(
                gamma=
                cfg["losses"]
                ["factorvae"]
                ["gamma"]
            )
        )

    def kl_loss(
        self,
        mu,
        logvar
    ):

        return (

            -0.5

            *

            torch.mean(

                1
                +
                logvar
                -
                mu.pow(2)
                -
                logvar.exp()
            )
        )

    def activation_flags(
        self
    ):

        stage_cfg = (
            self.cfg[
                "staged_activation"
            ]
        )

        if not stage_cfg["enabled"]:

            return {
                "phase": True,
                "kl": True,
                "orthogonality": True,
                "tc": True
            }

        progress = (

            self.current_epoch

            /

            max(
                1,
                self.total_epochs
            )
        )

        return {

            "phase":

                progress
                >=
                stage_cfg[
                    "phase_start"
                ],

            "kl":

                progress
                >=
                stage_cfg[
                    "kl_start"
                ],

            "orthogonality":

                progress
                >=
                stage_cfg[
                    "orthogonality_start"
                ],

            "tc":

                progress
                >=
                stage_cfg[
                    "tc_start"
                ]
        }

    def forward(

        self,

        outputs,

        targets,

        tc_logits=None
    ):

        flags = (
            self.activation_flags()
        )

        loss_dict = {}

        total = 0.0

        # ----------------------------------
        # Sync reconstruction stage state
        # ----------------------------------

        self.reconstruction.current_epoch = (
            self.current_epoch
        )

        self.reconstruction.total_epochs = (
            self.total_epochs
        )

        # ----------------------------------
        # Reconstruction
        # ----------------------------------

        recon_result = (

            self.reconstruction(

                outputs[
                    "reconstructions"
                ],

                targets
            )
        )

        recon_loss = (
            recon_result["loss"]
        )

        total += recon_loss

        loss_dict[
            "reconstruction"
        ] = recon_loss

        for k, v in recon_result.items():

            if k != "loss":

                loss_dict[k] = v

        # ----------------------------------
        # Multi Resolution
        # ----------------------------------

        mr_loss = self.multires(

            outputs[
                "reconstructions"
            ],

            targets
        )

        spectral_weight = (

            self.cfg["losses"]

            ["reconstruction"]

            ["spectral_weight"]
        )

        total += (

            spectral_weight

            *

            mr_loss
        )

        loss_dict[
            "multires"
        ] = mr_loss

        # ----------------------------------
        # KL
        # ----------------------------------

        if flags["kl"]:

            for latent_name in (

                outputs["mu"]
            ):

                mu = (
                    outputs["mu"]
                    [latent_name]
                )

                logvar = (
                    outputs["logvar"]
                    [latent_name]
                )

                kl = self.kl_loss(
                    mu,
                    logvar
                )

                beta = (

                    self.cfg["losses"]

                    ["kl"]

                    [latent_name]
                )

                total += (
                    beta
                    *
                    kl
                )

                loss_dict[
                    f"{latent_name}_kl"
                ] = kl

        else:

            for latent_name in (

                outputs["mu"]
            ):

                loss_dict[
                    f"{latent_name}_kl"
                ] = torch.tensor(
                    0.0,
                    device=total.device
                )

        # ----------------------------------
        # Orthogonality
        # ----------------------------------

        if flags["orthogonality"]:

            ortho = self.orthogonality(
                outputs["latents"]
            )

            ortho_weight = (

                self.cfg["losses"]

                ["orthogonality"]

                ["weight"]
            )

            total += (
                ortho_weight
                *
                ortho
            )

            loss_dict[
                "orthogonality"
            ] = ortho

        else:

            loss_dict[
                "orthogonality"
            ] = torch.tensor(
                0.0,
                device=total.device
            )

        # ----------------------------------
        # TC
        # ----------------------------------

        if (

            flags["tc"]

            and

            tc_logits is not None
        ):

            tc = self.factorvae(
                tc_logits
            )

            total += tc

            loss_dict[
                "tc"
            ] = tc

        else:

            loss_dict[
                "tc"
            ] = torch.tensor(
                0.0,
                device=total.device
            )

        # ----------------------------------
        # Final
        # ----------------------------------

        loss_dict[
            "total"
        ] = total

        return loss_dict