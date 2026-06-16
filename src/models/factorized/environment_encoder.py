# src/models/factorized/environment_encoder.py

import torch
import torch.nn as nn

from src.models.factorized.base_blocks import (
    ConvStem,
    DilatedTCN,
    TransformerStack
)

from src.models.factorized.pooling import (
    GlobalContextPooling
)

from src.models.factorized.encoder_utils import (
    SpectrogramToSequence
)

from src.models.factorized.variational_projection import (
    VariationalProjection
)


class EnvironmentBranch(nn.Module):

    def __init__(
        self,
        freq_bins,
        hidden_dim
    ):

        super().__init__()

        self.conv = ConvStem(

            in_channels=1,

            hidden_dim=hidden_dim
        )

        self.sequence = (

            SpectrogramToSequence(

                input_freq_bins=
                    freq_bins,

                hidden_dim=
                    hidden_dim
            )
        )

        self.tcn = (

            DilatedTCN(

                channels=
                    hidden_dim,

                layers=6
            )
        )

        self.transformer = (

            TransformerStack(

                dim=
                    hidden_dim,

                num_layers=2
            )
        )

        self.pool = (
            GlobalContextPooling(
                hidden_dim
            )
        )

    def forward(
        self,
        x
    ):

        x = self.conv(
            x
        )

        x = self.sequence(
            x
        )

        x = x.transpose(
            1,
            2
        )

        x = self.tcn(
            x
        )

        x = x.transpose(
            1,
            2
        )

        x = self.transformer(
            x
        )

        x = self.pool(
            x
        )

        return x


class EnvironmentEncoder(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        mcfg = cfg["model"]

        hidden_dim = (
            mcfg[
                "environment_encoder"
            ][
                "hidden_dim"
            ]
        )

        # ----------------------------------
        # Magnitude
        # strongest environment carrier
        # ----------------------------------

        self.magnitude_branch = (

            EnvironmentBranch(

                freq_bins=513,

                hidden_dim=
                    hidden_dim
            )
        )

        # ----------------------------------
        # MR1024
        # microphone + room response
        # ----------------------------------

        self.mr1024_branch = (

            EnvironmentBranch(

                freq_bins=513,

                hidden_dim=
                    hidden_dim
            )
        )

        # ----------------------------------
        # Instantaneous Frequency
        # strongest env separation
        # ----------------------------------

        self.if_branch = (

            EnvironmentBranch(

                freq_bins=513,

                hidden_dim=
                    hidden_dim
            )
        )

        self.feature_fusion = (

            nn.Sequential(

                nn.Linear(

                    hidden_dim * 3,

                    hidden_dim * 4
                ),

                nn.GELU(),

                nn.Dropout(
                    0.1
                ),

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
                        "environment"
                    ]
            )
        )

    def forward(
        self,
        feature_dict
    ):

        magnitude = (

            self.magnitude_branch(

                feature_dict[
                    "magnitude"
                ]
            )
        )

        mr1024 = (

            self.mr1024_branch(

                feature_dict[
                    "mr_mag_1024"
                ]
            )
        )

        if_feature = (

            self.if_branch(

                feature_dict[
                    "if"
                ]
            )
        )

        fused = torch.cat(

            [
                magnitude,
                mr1024,
                if_feature
            ],

            dim=-1
        )

        fused = self.feature_fusion(
            fused
        )

        return self.latent(
            fused
        )