# src/models/factorized/base_blocks.py

import torch
import torch.nn as nn

from torchaudio.models import Conformer


# --------------------------------------------------
# Conv Stem
# --------------------------------------------------

class ConvStem(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_dim
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_dim // 2,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                hidden_dim // 2
            ),

            nn.GELU(),

            nn.Conv2d(
                hidden_dim // 2,
                hidden_dim,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                hidden_dim
            ),

            nn.GELU()
        )

    def forward(
        self,
        x
    ):

        return self.net(
            x
        )


# --------------------------------------------------
# Feature Projector
# --------------------------------------------------

class FeatureProjector(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim
    ):

        super().__init__()

        self.proj = nn.Linear(
            input_dim,
            output_dim
        )

    def forward(
        self,
        x
    ):

        return self.proj(
            x
        )


# --------------------------------------------------
# Conformer Stack
# --------------------------------------------------

class ConformerStack(nn.Module):

    def __init__(
        self,
        dim,
        num_layers,
        num_heads=8,
        dropout=0.1
    ):

        super().__init__()

        self.conformer = Conformer(

            input_dim=dim,

            num_heads=num_heads,

            ffn_dim=dim * 4,

            num_layers=num_layers,

            depthwise_conv_kernel_size=31,

            dropout=dropout
        )

    def forward(
        self,
        x
    ):

        lengths = torch.full(

            (
                x.shape[0],
            ),

            x.shape[1],

            dtype=torch.long,

            device=x.device
        )

        x, _ = self.conformer(

            x,

            lengths
        )

        return x


# --------------------------------------------------
# Transformer Stack
# --------------------------------------------------

class TransformerStack(nn.Module):

    def __init__(
        self,
        dim,
        num_layers,
        num_heads=8,
        dropout=0.1
    ):

        super().__init__()

        layer = nn.TransformerEncoderLayer(

            d_model=dim,

            nhead=num_heads,

            dim_feedforward=
                dim * 4,

            dropout=dropout,

            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(

            layer,

            num_layers=num_layers
        )

    def forward(
        self,
        x
    ):

        return self.encoder(
            x
        )


# --------------------------------------------------
# Dilated TCN
# --------------------------------------------------

class DilatedTCN(nn.Module):

    def __init__(
        self,
        channels,
        layers=6,
        dropout=0.1
    ):

        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(layers):

            dilation = 2 ** i

            block = nn.Sequential(

                nn.Conv1d(

                    channels,

                    channels,

                    kernel_size=3,

                    padding=dilation,

                    dilation=dilation
                ),

                nn.BatchNorm1d(
                    channels
                ),

                nn.GELU(),

                nn.Dropout(
                    dropout
                )
            )

            self.blocks.append(
                block
            )

        self.norm = nn.LayerNorm(
            channels
        )

    def forward(
        self,
        x
    ):

        for block in self.blocks:

            residual = x

            x = block(x)

            x = x + residual

        x = x.transpose(
            1,
            2
        )

        x = self.norm(
            x
        )

        x = x.transpose(
            1,
            2
        )

        return x