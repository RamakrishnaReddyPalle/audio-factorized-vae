# src/models/factorized/factorized_vae.py

import torch
import torch.nn as nn

from src.models.factorized.content_encoder import (
    ContentEncoder
)

from src.models.factorized.speaker_encoder import (
    SpeakerEncoder
)

from src.models.factorized.environment_encoder import (
    EnvironmentEncoder
)

from src.models.factorized.excitation_encoder import (
    ExcitationEncoder
)

from src.models.factorized.fidelity_encoder import (
    FidelityEncoder
)

from src.models.factorized.factorized_decoder_core import (
    FactorizedDecoderCore
)

from src.models.factorized.factorized_decoder import (
    FactorizedDecoder
)

from src.models.factorized.feature_group_manager import (
    FeatureGroupManager
)


class FactorizedVAE(
    nn.Module
):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.group_manager = (
            FeatureGroupManager()
        )

        self.content_encoder = (
            ContentEncoder(cfg)
        )

        self.speaker_encoder = (
            SpeakerEncoder(cfg)
        )

        self.environment_encoder = (
            EnvironmentEncoder(cfg)
        )

        self.excitation_encoder = (
            ExcitationEncoder(cfg)
        )

        self.fidelity_encoder = (
            FidelityEncoder(cfg)
        )

        self.decoder_core = (
            FactorizedDecoderCore(cfg)
        )

        self.decoder = (
            FactorizedDecoder(cfg)
        )

    def encode(
        self,
        feature_groups
    ):

        latents = {}
        mu = {}
        logvar = {}

        content_out = self.content_encoder(
            feature_groups["content"]
        )

        latents["content"] = content_out["z"]
        mu["content"] = content_out["mu"]
        logvar["content"] = content_out["logvar"]

        speaker_out = self.speaker_encoder(
            feature_groups["speaker"]
        )

        latents["speaker"] = speaker_out["z"]
        mu["speaker"] = speaker_out["mu"]
        logvar["speaker"] = speaker_out["logvar"]

        environment_out = self.environment_encoder(
            feature_groups["environment"]
        )

        latents["environment"] = environment_out["z"]
        mu["environment"] = environment_out["mu"]
        logvar["environment"] = environment_out["logvar"]

        excitation_out = self.excitation_encoder(
            feature_groups["excitation"]
        )

        latents["excitation"] = excitation_out["z"]
        mu["excitation"] = excitation_out["mu"]
        logvar["excitation"] = excitation_out["logvar"]

        fidelity_out = self.fidelity_encoder(
            feature_groups["fidelity"]
        )

        latents["fidelity"] = fidelity_out["z"]
        mu["fidelity"] = fidelity_out["mu"]
        logvar["fidelity"] = fidelity_out["logvar"]

        return (
            latents,
            mu,
            logvar
        )

    def build_joint_latent(
        self,
        latents
    ):

        return torch.cat(
            [
                latents["content"],
                latents["speaker"],
                latents["environment"],
                latents["excitation"],
                latents["fidelity"]
            ],
            dim=-1
        )

    def decode(

        self,

        latents,

        target_lengths
    ):

        hidden = self.decoder_core(

            latents["content"],

            latents["speaker"],

            latents["environment"],

            latents["excitation"],

            latents["fidelity"],

            target_lengths["logmel"]
        )

        return self.decoder(

            hidden,

            target_lengths
        )

    def forward(
        self,
        batch
    ):

        feature_groups = (

            self.group_manager
            .build_groups(batch)
        )

        latents, mu, logvar = (

            self.encode(
                feature_groups
            )
        )

        joint_latent = (

            self.build_joint_latent(
                latents
            )
        )

        target_lengths = {

            "logmel":
                batch["logmel"].shape[-1],

            "mr_mag_256":
                batch["mr_mag_256"].shape[-1],

            "mr_mag_512":
                batch["mr_mag_512"].shape[-1],

            "magnitude":
                batch["magnitude"].shape[-1],

            "mr_mag_1024":
                batch["mr_mag_1024"].shape[-1]
        }

        reconstructions = (

            self.decode(

                latents,

                target_lengths
            )
        )

        return {

            "groups":
                feature_groups,

            "latents":
                latents,

            "mu":
                mu,

            "logvar":
                logvar,

            "reconstructions":
                reconstructions,

            "joint_latent":
                joint_latent
        }