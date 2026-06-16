# src/models/factorized/identity_backbone.py

import torch
import torch.nn as nn


class IdentityBackbone(nn.Module):

    def __init__(
        self,
        content_dim,
        speaker_dim,
        hidden_dim,
        num_layers=4,
        num_heads=8
    ):

        super().__init__()

        self.input_proj = nn.Linear(

            content_dim + speaker_dim,

            hidden_dim
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model=
                    hidden_dim,

                nhead=
                    num_heads,

                batch_first=
                    True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(

                encoder_layer,

                num_layers=
                    num_layers
            )
        )

    def forward(

        self,

        z_content,

        z_speaker,

        target_length
    ):

        x = torch.cat(

            [
                z_content,
                z_speaker
            ],

            dim=-1
        )

        x = self.input_proj(
            x
        )

        # ----------------------------------
        # [B,D]
        # ->
        # [B,T,D]
        # ----------------------------------

        x = x.unsqueeze(1).expand(

            -1,

            target_length,

            -1
        )

        x = self.transformer(
            x
        )

        return x