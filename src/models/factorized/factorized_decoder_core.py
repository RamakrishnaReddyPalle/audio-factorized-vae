# src/models/factorized/factorized_decoder_core.py

import torch.nn as nn

from src.models.factorized.identity_backbone import (
    IdentityBackbone
)

from src.models.factorized.environment_film import (
    EnvironmentFiLM
)

from src.models.factorized.excitation_cross_attention import (
    ExcitationCrossAttention
)

from src.models.factorized.fidelity_cross_attention import (
    FidelityCrossAttention
)


class FactorizedDecoderCore(nn.Module):

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

        latent_cfg = (
            cfg["model"][
                "latent_dims"
            ]
        )

        self.identity = IdentityBackbone(

            content_dim=
                latent_cfg["content"],

            speaker_dim=
                latent_cfg["speaker"],

            hidden_dim=
                hidden_dim
        )

        self.environment_film = EnvironmentFiLM(

            env_dim=
                latent_cfg["environment"],

            hidden_dim=
                hidden_dim
        )

        self.excitation_attention = (
            ExcitationCrossAttention(

                hidden_dim=
                    hidden_dim,

                excitation_dim=
                    latent_cfg["excitation"]
            )
        )

        self.fidelity_attention = (
            FidelityCrossAttention(

                hidden_dim=
                    hidden_dim,

                fidelity_dim=
                    latent_cfg["fidelity"]
            )
        )

    def forward(

        self,

        z_content,

        z_speaker,

        z_environment,

        z_excitation,

        z_fidelity,

        target_length
    ):

        x = self.identity(

            z_content,

            z_speaker,

            target_length
        )

        x = self.environment_film(

            x,

            z_environment
        )

        x = self.excitation_attention(

            x,

            z_excitation
        )

        x = self.fidelity_attention(

            x,

            z_fidelity
        )

        return x