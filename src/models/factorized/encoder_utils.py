# src/models/factorized/encoder_utils.py

import torch
import torch.nn as nn


class SpectrogramToSequence(nn.Module):

    """
    Input:
        [B,C,F,T]

    Output:
        [B,T,H]
    """

    def __init__(
        self,
        input_freq_bins,
        hidden_dim
    ):

        super().__init__()

        self.projection = nn.Linear(

            input_freq_bins,

            hidden_dim
        )

    def forward(
        self,
        x
    ):

        # [B,C,F,T]

        x = x.mean(
            dim=1
        )

        # [B,F,T]

        x = x.transpose(
            1,
            2
        )

        # [B,T,F]

        x = self.projection(
            x
        )

        # [B,T,H]

        return x