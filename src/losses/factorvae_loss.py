# src/losses/factorvae_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorVAELoss(
    nn.Module
):

    def __init__(
        self,
        gamma
    ):

        super().__init__()

        self.gamma = gamma

        # ----------------------------------
        # Injected from TotalLoss
        # ----------------------------------

        self.current_epoch = 0
        self.total_epochs = 1

    def discriminator_loss(

        self,

        real_logits,

        permuted_logits
    ):

        real_labels = torch.ones(

            real_logits.shape[0],

            dtype=torch.long,

            device=real_logits.device
        )

        perm_labels = torch.zeros(

            permuted_logits.shape[0],

            dtype=torch.long,

            device=permuted_logits.device
        )

        real_loss = F.cross_entropy(

            real_logits,

            real_labels
        )

        perm_loss = F.cross_entropy(

            permuted_logits,

            perm_labels
        )

        return (

            real_loss

            +

            perm_loss

        ) * 0.5

    def tc_estimate(

        self,

        real_logits
    ):

        return (

            real_logits[:, 0]

            -

            real_logits[:, 1]

        ).mean()

    def tc_scale(
        self
    ):

        progress = (

            self.current_epoch

            /

            max(
                1,
                self.total_epochs
            )
        )

        return min(

            1.0,

            progress / 0.70
        )

    def forward(

        self,

        real_logits,

        permuted_logits=None
    ):

        tc_est = self.tc_estimate(

            real_logits
        )

        scale = (
            self.tc_scale()
        )

        tc_loss = (

            self.gamma

            *

            scale

            *

            tc_est
        )

        result = {

            "tc_loss":
                tc_loss,

            "tc_estimate":
                tc_est.detach(),

            "tc_scale":
                torch.tensor(
                    float(scale),
                    device=real_logits.device
                )
        }

        if permuted_logits is not None:

            result[
                "discriminator_loss"
            ] = self.discriminator_loss(

                real_logits,

                permuted_logits
            )

        return result