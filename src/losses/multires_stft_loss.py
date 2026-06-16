# src/losses/multires_stft_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionFeatureLoss(
    nn.Module
):

    def __init__(
        self
    ):
        super().__init__()

    def align_target(
        self,
        prediction,
        target
    ):

        # ----------------------------------
        # Dataset tensors may arrive as
        # [B,1,F,T]
        #
        # Decoder outputs are
        # [B,F,T]
        # ----------------------------------

        if target.ndim == 4:

            target = target.squeeze(1)

        pred_t = prediction.shape[-1]

        target_t = target.shape[-1]

        # ----------------------------------
        # Decoder currently reconstructs
        # fixed temporal length (32)
        #
        # Align target to prediction
        # ----------------------------------

        if pred_t != target_t:

            target = F.interpolate(

                target,

                size=pred_t,

                mode="linear",

                align_corners=False
            )

        return target

    def single_loss(
        self,
        prediction,
        target
    ):

        target = self.align_target(

            prediction,

            target
        )

        return (

            F.l1_loss(
                prediction,
                target
            )

            +

            F.mse_loss(
                prediction,
                target
            )
        )

    def forward(
        self,
        predictions,
        targets
    ):

        loss = 0.0

        loss += self.single_loss(

            predictions["mr_mag_256"],

            targets["mr_mag_256"]
        )

        loss += self.single_loss(

            predictions["mr_mag_512"],

            targets["mr_mag_512"]
        )

        loss += self.single_loss(

            predictions["mr_mag_1024"],

            targets["mr_mag_1024"]
        )

        return loss