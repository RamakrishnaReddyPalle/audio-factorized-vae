# src/models/factorized/reconstruction_heads.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.factorized.feature_heads import (
    FeatureHead,
    MagnitudeHead,
    IFHead,
    PhaseHead
)


class ReconstructionHeads(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        hidden_dim = (
            cfg["factorized_model"][
                "transformer_dim"
            ]
        )

        self.logmel = MagnitudeHead(

            hidden_dim=hidden_dim,
            output_freq_bins=80
        )

        self.mr_mag_256 = MagnitudeHead(

            hidden_dim=hidden_dim,
            output_freq_bins=129
        )

        self.mr_mag_512 = MagnitudeHead(

            hidden_dim=hidden_dim,
            output_freq_bins=257
        )

        self.magnitude = MagnitudeHead(

            hidden_dim=hidden_dim,
            output_freq_bins=513
        )

        self.mr_mag_1024 = MagnitudeHead(

            hidden_dim=hidden_dim,
            output_freq_bins=513
        )

        self.if_head = IFHead(

            hidden_dim=hidden_dim,
            output_freq_bins=513
        )

        self.modgd = FeatureHead(

            hidden_dim=hidden_dim,
            output_freq_bins=513
        )

        self.phase_sin = PhaseHead(

            hidden_dim=hidden_dim,
            output_freq_bins=513
        )

        self.phase_cos = PhaseHead(

            hidden_dim=hidden_dim,
            output_freq_bins=513
        )

    def resize_hidden(

        self,

        hidden,

        target_length
    ):

        if hidden.shape[1] == target_length:

            return hidden

        hidden = hidden.transpose(
            1,
            2
        )

        hidden = F.interpolate(

            hidden,

            size=target_length,

            mode="linear",

            align_corners=False
        )

        hidden = hidden.transpose(
            1,
            2
        )

        return hidden

    def forward(

        self,

        hidden,

        target_lengths
    ):

        h_logmel = self.resize_hidden(

            hidden,

            target_lengths["logmel"]
        )

        h_mr256 = self.resize_hidden(

            hidden,

            target_lengths["mr_mag_256"]
        )

        h_mr512 = self.resize_hidden(

            hidden,

            target_lengths["mr_mag_512"]
        )

        h_mag = self.resize_hidden(

            hidden,

            target_lengths["magnitude"]
        )

        h_mr1024 = self.resize_hidden(

            hidden,

            target_lengths["mr_mag_1024"]
        )

        outputs = {

            "logmel":
                self.logmel(
                    h_logmel
                ),

            "mr_mag_256":
                self.mr_mag_256(
                    h_mr256
                ),

            "mr_mag_512":
                self.mr_mag_512(
                    h_mr512
                ),

            "magnitude":
                self.magnitude(
                    h_mag
                ),

            "mr_mag_1024":
                self.mr_mag_1024(
                    h_mr1024
                ),

            "if":
                self.if_head(
                    h_mag
                ),

            "modgd":
                self.modgd(
                    h_mag
                ),

            "phase_sin":
                self.phase_sin(
                    h_mag
                ),

            "phase_cos":
                self.phase_cos(
                    h_mag
                )
        }

        return outputs