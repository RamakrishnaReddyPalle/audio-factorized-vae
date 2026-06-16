# src/models/factorized/fidelity_encoder.py

import torch
import torch.nn as nn

from src.models.factorized.multi_resolution_conv import (
    MultiResolutionConv
)

from src.models.factorized.cross_scale_attention import (
    CrossScaleAttention
)

from src.models.factorized.base_blocks import (
    TransformerStack
)

from src.models.factorized.encoder_utils import (
    SpectrogramToSequence
)

from src.models.factorized.pooling import (
    TemporalAttentionPooling
)

from src.models.factorized.variational_projection import (
    VariationalProjection
)


class FidelityBranch(nn.Module):

    def __init__(
        self,
        freq_bins,
        hidden_dim
    ):

        super().__init__()

        self.multi_res = (

            MultiResolutionConv(

                in_channels=1,

                hidden_dim=hidden_dim
            )
        )

        self.sequence = (

            SpectrogramToSequence(

                input_freq_bins=
                    freq_bins,

                hidden_dim=
                    hidden_dim
            )
        )

        self.cross_scale = (

            CrossScaleAttention(

                dim=hidden_dim,

                heads=8
            )
        )

        self.transformer = (

            TransformerStack(

                dim=hidden_dim,

                num_layers=2
            )
        )

        self.pool = (

            TemporalAttentionPooling(
                hidden_dim
            )
        )

    def forward(
        self,
        x
    ):

        x = self.multi_res(
            x
        )

        x = self.sequence(
            x
        )

        x = self.cross_scale(
            x
        )

        x = self.transformer(
            x
        )

        x = self.pool(
            x
        )

        return x


class FidelityEncoder(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        mcfg = cfg["model"]

        hidden_dim = (
            mcfg["fidelity_encoder"][
                "hidden_dim"
            ]
        )

        # -------------------------
        # Fidelity branches
        # -------------------------

        self.phase_sin_branch = (

            FidelityBranch(

                freq_bins=513,

                hidden_dim=hidden_dim
            )
        )

        self.phase_cos_branch = (

            FidelityBranch(

                freq_bins=513,

                hidden_dim=hidden_dim
            )
        )

        self.mr512_branch = (

            FidelityBranch(

                freq_bins=257,

                hidden_dim=hidden_dim
            )
        )

        self.mr1024_branch = (

            FidelityBranch(

                freq_bins=513,

                hidden_dim=hidden_dim
            )
        )

        self.magnitude_branch = (

            FidelityBranch(

                freq_bins=513,

                hidden_dim=hidden_dim
            )
        )

        self.modgd_branch = (

            FidelityBranch(

                freq_bins=513,

                hidden_dim=hidden_dim
            )
        )

        # -------------------------
        # Fusion
        # -------------------------

        self.feature_fusion = (

            nn.Sequential(

                nn.Linear(

                    hidden_dim * 6,

                    hidden_dim * 4
                ),

                nn.GELU(),

                nn.Dropout(0.1),

                nn.Linear(

                    hidden_dim * 4,

                    hidden_dim * 2
                ),

                nn.GELU(),

                nn.Linear(

                    hidden_dim * 2,

                    hidden_dim
                )
            )
        )

        self.latent = (

            VariationalProjection(

                input_dim=
                    hidden_dim,

                latent_dim=
                    mcfg[
                        "latent_dims"
                    ][
                        "fidelity"
                    ]
            )
        )

    def forward(
        self,
        feature_dict
    ):

        phase_sin = self.phase_sin_branch(
            feature_dict["phase_sin"]
        )

        phase_cos = self.phase_cos_branch(
            feature_dict["phase_cos"]
        )

        mr512 = self.mr512_branch(
            feature_dict["mr_mag_512"]
        )

        mr1024 = self.mr1024_branch(
            feature_dict["mr_mag_1024"]
        )

        magnitude = self.magnitude_branch(
            feature_dict["magnitude"]
        )

        modgd = self.modgd_branch(
            feature_dict["modgd"]
        )

        fused = torch.cat(

            [
                phase_sin,
                phase_cos,
                mr512,
                mr1024,
                magnitude,
                modgd
            ],

            dim=-1
        )

        fused = self.feature_fusion(
            fused
        )

        return self.latent(
            fused
        )