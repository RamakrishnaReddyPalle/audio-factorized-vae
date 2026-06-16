# src/models/factorized/factorized_decoder.py

import torch.nn as nn

from src.models.factorized.reconstruction_heads import (
    ReconstructionHeads
)


class FactorizedDecoder(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.heads = (
            ReconstructionHeads(
                cfg
            )
        )

    def forward(

        self,

        hidden_sequence,

        target_lengths
    ):

        return self.heads(

            hidden_sequence,

            target_lengths
        )