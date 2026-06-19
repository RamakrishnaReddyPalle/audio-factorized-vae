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

        self.last_flags = None

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

        targets
    ):

        flags = (
            self.activation_flags()
        )

        progress = (

            self.current_epoch

            /

            max(
                1,
                self.total_epochs
            )
        )

        if flags != self.last_flags:

            print()
            print("=" * 70)
            print("[LOSS ACTIVATION STATE]")
            print("=" * 70)

            print(
                f"epoch={self.current_epoch}"
            )

            print(
                f"progress={progress:.4f}"
            )

            print(
                f"phase={flags['phase']}"
            )

            print(
                f"kl={flags['kl']}"
            )

            print(
                f"orthogonality={flags['orthogonality']}"
            )

            print(
                f"tc={flags['tc']}"
            )

            self.last_flags = (
                flags.copy()
            )

        loss_dict = {}

        total = 0.0

        # ----------------------------------
        # Sync reconstruction state
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

        kl_total = torch.tensor(
            0.0,
            device=recon_loss.device
        )

        if flags["kl"]:

            for latent_name in outputs["mu"]:

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

                kl_total += (
                    beta * kl
                )

                total += (
                    beta * kl
                )

                loss_dict[
                    f"{latent_name}_kl"
                ] = kl

                loss_dict[
                    f"{latent_name}_beta"
                ] = torch.tensor(
                    float(beta),
                    device=kl.device
                )

            if self.current_epoch % 10 == 0:

                print()
                print(
                    "[KL DIAGNOSTICS]"
                )

                for latent_name in outputs["mu"]:

                    print(
                        latent_name,
                        f"KL={loss_dict[f'{latent_name}_kl'].item():.6f}",
                        f"BETA={loss_dict[f'{latent_name}_beta'].item():.6f}"
                    )

        else:

            for latent_name in outputs["mu"]:

                loss_dict[
                    f"{latent_name}_kl"
                ] = torch.tensor(
                    0.0,
                    device=recon_loss.device
                )

        loss_dict[
            "kl_total"
        ] = kl_total

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

            if self.current_epoch % 10 == 0:

                print()

                print(
                    "[ORTHOGONALITY]"
                )

                print(
                    f"loss={ortho.item():.6f}"
                )

        else:

            loss_dict[
                "orthogonality"
            ] = torch.tensor(
                0.0,
                device=recon_loss.device
            )

        # ----------------------------------
        # Total Correlation
        # ----------------------------------

        tc_logits = outputs.get(
            "tc_logits",
            None
        )

        tc_logits_permuted = outputs.get(
            "tc_logits_permuted",
            None
        )

        if (

            flags["tc"]

            and

            tc_logits is not None

            and

            tc_logits_permuted is not None
        ):

            tc_result = (

                self.factorvae(

                    tc_logits,

                    tc_logits_permuted
                )
            )

            tc = tc_result[
                "tc_loss"
            ]

            total += tc

            loss_dict[
                "tc"
            ] = tc

            if (
                "discriminator_loss"
                in tc_result
            ):

                loss_dict[
                    "discriminator_loss"
                ] = tc_result[
                    "discriminator_loss"
                ]

            if self.current_epoch % 10 == 0:

                print()

                print(
                    "[TC DIAGNOSTICS]"
                )

                print(
                    f"tc={tc.item():.6f}"
                )

                if (
                    "discriminator_loss"
                    in tc_result
                ):

                    print(
                        f"disc={tc_result['discriminator_loss'].item():.6f}"
                    )

        else:

            loss_dict[
                "tc"
            ] = torch.tensor(
                0.0,
                device=recon_loss.device
            )

        # ----------------------------------
        # Final
        # ----------------------------------

        loss_dict[
            "total"
        ] = total

        return loss_dict