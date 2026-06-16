# src/models/factorized/excitation_encoder.py

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


class ExcitationEncoder(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        mcfg = cfg["model"]

        hidden_dim = (
            mcfg[
                "excitation_encoder"
            ][
                "hidden_dim"
            ]
        )

        self.conv_stem = ConvStem(

            in_channels=1,

            hidden_dim=hidden_dim
        )

        self.sequence = SpectrogramToSequence(

            input_freq_bins=513,

            hidden_dim=hidden_dim
        )

        self.conformer = ConformerStack(

            dim=hidden_dim,

            num_layers=3
        )

        self.pool = TemporalAttentionPooling(
            hidden_dim
        )

        self.latent = VariationalProjection(

            input_dim=hidden_dim,

            latent_dim=
                mcfg[
                    "latent_dims"
                ][
                    "excitation"
                ]
        )

    def forward(
        self,
        feature_dict
    ):

        x = feature_dict[
            "modgd"
        ]

        x = self.conv_stem(
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

        return self.latent(
            x
        )