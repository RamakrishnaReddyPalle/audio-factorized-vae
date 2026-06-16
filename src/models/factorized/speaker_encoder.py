# src/models/factorized/speaker_encoder.py

import torch
import torch.nn as nn

from src.models.factorized.base_blocks import (
    ConvStem,
    ConformerStack,
    TransformerStack
)

from src.models.factorized.pooling import (
    AttentiveStatsPooling
)

from src.models.factorized.encoder_utils import (
    SpectrogramToSequence
)

from src.models.factorized.variational_projection import (
    VariationalProjection
)


class SpeakerBranch(nn.Module):

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

                num_layers=4
            )
        )

        self.transformer = (

            TransformerStack(

                dim=
                    hidden_dim,

                num_layers=1
            )
        )

        self.pool = (
            AttentiveStatsPooling(
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

        x = self.transformer(
            x
        )

        x = self.pool(
            x
        )

        return x


class SpeakerEncoder(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        mcfg = cfg["model"]

        hidden_dim = (
            mcfg[
                "speaker_encoder"
            ][
                "hidden_dim"
            ]
        )

        # ----------------------------------
        # MR512
        # strongest speaker cue
        # ----------------------------------

        self.mr512_branch = (

            SpeakerBranch(

                freq_bins=257,

                hidden_dim=
                    hidden_dim
            )
        )

        # ----------------------------------
        # MR256
        # temporal cue
        # ----------------------------------

        self.mr256_branch = (

            SpeakerBranch(

                freq_bins=129,

                hidden_dim=
                    hidden_dim
            )
        )

        # ----------------------------------
        # LogMel
        # phonetic speaker cue
        # ----------------------------------

        self.logmel_branch = (

            SpeakerBranch(

                freq_bins=80,

                hidden_dim=
                    hidden_dim
            )
        )

        pooled_dim = (
            hidden_dim * 2
        )

        self.feature_fusion = (

            nn.Sequential(

                nn.Linear(

                    pooled_dim * 3,

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
                        "speaker"
                    ]
            )
        )

    def forward(
        self,
        feature_dict
    ):

        mr512 = (
            self.mr512_branch(

                feature_dict[
                    "mr_mag_512"
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

        logmel = (
            self.logmel_branch(

                feature_dict[
                    "logmel"
                ]
            )
        )

        fused = torch.cat(

            [
                mr512,
                mr256,
                logmel
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