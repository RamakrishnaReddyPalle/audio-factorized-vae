# src/trainers/factorvae_scheduler.py

from pathlib import Path

import torch
import torch.nn as nn

from src.utils.config_loader import (
    load_yaml
)

from src.models.factorized.factorvae_discriminator import (
    FactorVAEDiscriminator
)

from src.models.factorized.factor_utils import (
    permute_dims
)


class FactorVAEScheduler:

    def __init__(
        self,
        project_root,
        model,
        device
    ):

        self.project_root = Path(
            project_root
        )

        self.device = device

        self.model = model

        model_cfg = load_yaml(

            self.project_root
            /
            "configs"
            /
            "model_config.yaml"
        )

        latent_cfg = (
            model_cfg["model"]
            ["latent_dims"]
        )

        latent_dim = (

            latent_cfg["content"]

            +

            latent_cfg["speaker"]

            +

            latent_cfg["environment"]

            +

            latent_cfg["excitation"]

            +

            latent_cfg["fidelity"]
        )

        self.discriminator = (
            FactorVAEDiscriminator(
                latent_dim=latent_dim
            ).to(device)
        )

        self.optimizer = (
            torch.optim.Adam(

                self.discriminator.parameters(),

                lr=1e-4,

                betas=(0.5, 0.9)
            )
        )

        self.ce_loss = (
            nn.CrossEntropyLoss()
        )

    def build_targets(

        self,

        batch_size,

        real=True
    ):

        label = 1 if real else 0

        return torch.full(

            (batch_size,),

            label,

            dtype=torch.long,

            device=self.device
        )

    def discriminator_step(

        self,

        joint_latent
    ):

        z_real = (
            joint_latent.detach()
        )

        z_perm = (
            permute_dims(
                z_real
            )
        )

        real_logits = (
            self.discriminator(
                z_real
            )
        )

        perm_logits = (
            self.discriminator(
                z_perm
            )
        )

        real_targets = (
            self.build_targets(
                z_real.size(0),
                real=True
            )
        )

        perm_targets = (
            self.build_targets(
                z_perm.size(0),
                real=False
            )
        )

        loss_real = self.ce_loss(
            real_logits,
            real_targets
        )

        loss_perm = self.ce_loss(
            perm_logits,
            perm_targets
        )

        disc_loss = (
            loss_real
            +
            loss_perm
        ) * 0.5

        self.optimizer.zero_grad(
            set_to_none=True
        )

        disc_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            5.0
        )

        self.optimizer.step()

        with torch.no_grad():

            real_acc = (

                real_logits.argmax(dim=1)

                ==

                real_targets

            ).float().mean()

            perm_acc = (

                perm_logits.argmax(dim=1)

                ==

                perm_targets

            ).float().mean()

        return {

            "disc_loss":
                disc_loss.detach(),

            "disc_real_acc":
                real_acc.detach(),

            "disc_perm_acc":
                perm_acc.detach()
        }

    def generator_logits(

        self,

        joint_latent
    ):

        z_perm = (
            permute_dims(
                joint_latent
            )
        )

        return {

            "real":
                self.discriminator(
                    joint_latent
                ),

            "permuted":
                self.discriminator(
                    z_perm
                )
        }

    def state_dict(
        self
    ):

        return {

            "discriminator":
                self.discriminator.state_dict(),

            "optimizer":
                self.optimizer.state_dict()
        }

    def load_state_dict(
        self,
        state
    ):

        if "discriminator" in state:

            self.discriminator.load_state_dict(
                state["discriminator"]
            )

        if "optimizer" in state:

            self.optimizer.load_state_dict(
                state["optimizer"]
            )