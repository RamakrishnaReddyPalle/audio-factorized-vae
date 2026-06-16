# src/models/factorized/pooling.py

import torch
import torch.nn as nn


# --------------------------------------------------
# Temporal Attention Pooling
# --------------------------------------------------

class TemporalAttentionPooling(nn.Module):

    def __init__(
        self,
        dim,
        dropout=0.1
    ):

        super().__init__()

        self.norm = nn.LayerNorm(
            dim
        )

        self.attention = nn.Sequential(

            nn.Linear(
                dim,
                dim
            ),

            nn.Tanh(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                dim,
                1
            )
        )

    def forward(
        self,
        x
    ):

        x_norm = self.norm(
            x
        )

        weights = self.attention(
            x_norm
        )

        weights = torch.softmax(

            weights,

            dim=1
        )

        pooled = torch.sum(

            x * weights,

            dim=1
        )

        return pooled


# --------------------------------------------------
# Attentive Statistics Pooling
# --------------------------------------------------

class AttentiveStatsPooling(nn.Module):

    def __init__(
        self,
        dim,
        dropout=0.1
    ):

        super().__init__()

        self.norm = nn.LayerNorm(
            dim
        )

        self.attention = nn.Sequential(

            nn.Linear(
                dim,
                dim
            ),

            nn.Tanh(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                dim,
                1
            )
        )

    def forward(
        self,
        x
    ):

        x_norm = self.norm(
            x
        )

        w = self.attention(
            x_norm
        )

        w = torch.softmax(
            w,
            dim=1
        )

        mean = torch.sum(
            x * w,
            dim=1
        )

        var = torch.sum(

            ((x - mean.unsqueeze(1)) ** 2)
            * w,

            dim=1
        )

        std = torch.sqrt(

            torch.clamp(
                var,
                min=1e-6
            )
        )

        return torch.cat(

            [
                mean,
                std
            ],

            dim=-1
        )


# --------------------------------------------------
# Global Context Pooling
# --------------------------------------------------

class GlobalContextPooling(nn.Module):

    def __init__(
        self,
        dim
    ):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(
                dim,
                dim
            ),

            nn.GELU(),

            nn.Linear(
                dim,
                dim
            )
        )

    def forward(
        self,
        x
    ):

        mean = x.mean(
            dim=1
        )

        context = self.fc(
            mean
        )

        return context + mean