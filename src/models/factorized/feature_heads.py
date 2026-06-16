# src/models/factorized/feature_heads.py

import torch
import torch.nn as nn


class FeatureHead(nn.Module):

    """
    Generic reconstruction head.

    Input:
        [B,T,D]

    Output:
        [B,F,T]
    """

    def __init__(
        self,
        hidden_dim,
        output_freq_bins,
        dropout=0.1
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                output_freq_bins
            )
        )

    def forward(
        self,
        x
    ):

        x = self.net(
            x
        )

        return x.permute(
            0,
            2,
            1
        )


class MagnitudeHead(nn.Module):

    """
    Magnitude-like outputs.

    Positive bounded outputs.

    Output:
        [B,F,T]
    """

    def __init__(
        self,
        hidden_dim,
        output_freq_bins,
        dropout=0.1
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                output_freq_bins
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        x
    ):

        x = self.net(
            x
        )

        return x.permute(
            0,
            2,
            1
        )


class PhaseHead(nn.Module):

    """
    Phase decoder.

    Produces:

        sin(phi)
        cos(phi)

    bounded [-1,1]
    """

    def __init__(
        self,
        hidden_dim,
        output_freq_bins=513,
        dropout=0.1
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                output_freq_bins
            ),

            nn.Tanh()
        )

    def forward(
        self,
        x
    ):

        x = self.net(
            x
        )

        return x.permute(
            0,
            2,
            1
        )


class IFHead(nn.Module):

    """
    IF reconstruction.

    Output is intentionally
    unbounded because
    normalization will happen
    inside the loss function.
    """

    def __init__(
        self,
        hidden_dim,
        output_freq_bins=513,
        dropout=0.1
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                output_freq_bins
            )
        )

    def forward(
        self,
        x
    ):

        x = self.net(
            x
        )

        return x.permute(
            0,
            2,
            1
        )