# src/models/factorized/multi_resolution_conv.py

import torch
import torch.nn as nn


class MultiResolutionConv(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_dim
    ):

        super().__init__()

        self.branch3 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_dim,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                hidden_dim
            ),

            nn.GELU()
        )

        self.branch5 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_dim,
                5,
                padding=2
            ),

            nn.BatchNorm2d(
                hidden_dim
            ),

            nn.GELU()
        )

        self.branch7 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_dim,
                7,
                padding=3
            ),

            nn.BatchNorm2d(
                hidden_dim
            ),

            nn.GELU()
        )

        self.fusion = nn.Sequential(

            nn.Conv2d(

                hidden_dim * 3,

                hidden_dim,

                kernel_size=1
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

        b3 = self.branch3(x)

        b5 = self.branch5(x)

        b7 = self.branch7(x)

        x = torch.cat(

            [b3, b5, b7],

            dim=1
        )

        x = self.fusion(x)

        return x