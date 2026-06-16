# src/models/factorized/content_encoder.py

import torch
import torch.nn as nn

from src.models.factorized.base_blocks import (
    ConvStem,
    ConformerStack
)

from src.models.factorized.pooling import (
    TemporalAttentionPooling
)

from src.models.factorized.encoder_utils import (
    SpectrogramToSequence
)

from src.models.factorized.variational_projection import (
    VariationalProjection
)


class ContentBranch(nn.Module):

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

        self.conformer = (

            ConformerStack(

                dim=
                    hidden_dim,

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

        x = self.conv(
            x
        )

        x = self.sequence(
            x
        )

        x = self.conformer(
            x
        )

        x = self.pool(
            x
        )

        return x


class ContentEncoder(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        mcfg = cfg["model"]

        hidden_dim = (
            mcfg[
                "content_encoder"
            ][
                "hidden_dim"
            ]
        )

        self.logmel_branch = (

            ContentBranch(

                freq_bins=80,

                hidden_dim=
                    hidden_dim
            )
        )

        self.mr256_branch = (

            ContentBranch(

                freq_bins=129,

                hidden_dim=
                    hidden_dim
            )
        )

        self.if_branch = (

            ContentBranch(

                freq_bins=513,

                hidden_dim=
                    hidden_dim
            )
        )

        self.feature_fusion = (

            nn.Sequential(

                nn.Linear(

                    hidden_dim * 3,

                    hidden_dim * 2
                ),

                nn.GELU(),

                nn.Dropout(
                    0.1
                ),

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
                        "content"
                    ]
            )
        )

    def forward(
        self,
        feature_dict
    ):

        logmel = (
            self.logmel_branch(
                feature_dict[
                    "logmel"
                ]
            )
        )

        mr256 = (
            self.mr256_branch(
                feature_dict[
                    "mr_mag_256"
                ]
            )
        )

        if_feat = (
            self.if_branch(
                feature_dict[
                    "if"
                ]
            )
        )

        fused = torch.cat(

            [
                logmel,
                mr256,
                if_feat
            ],

            dim=-1
        )

        fused = (
            self.feature_fusion(
                fused
            )
        )

        return self.latent(
            fused
        )