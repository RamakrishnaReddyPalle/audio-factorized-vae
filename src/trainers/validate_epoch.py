# src/trainers/validate_epoch.py

import torch

from torch.cuda.amp import (
    autocast
)


def validate_epoch(

    model,

    loader,

    loss_fn,

    metrics_tracker,

    latent_monitor,

    factorvae_scheduler,

    device,

    cfg,

    epoch
):

    model.eval()

    metrics_tracker.reset()

    last_latent_report = {}

    amp_enabled = (

        cfg["mixed_precision"]
        ["enabled"]
    )

    with torch.no_grad():

        for batch in loader:

            batch = {

                k: (
                    v.to(device)

                    if torch.is_tensor(v)

                    else v
                )

                for k, v in batch.items()
            }

            with autocast(

                enabled=amp_enabled
            ):

                outputs = model(
                    batch
                )

                # ----------------------------------
                # TC logits from current
                # discriminator state
                # ----------------------------------

                tc_logits = None
                tc_logits_permuted = None

                if (

                    factorvae_scheduler
                    is not None
                ):

                    tc_outputs = (

                        factorvae_scheduler
                        .generator_logits(

                            outputs[
                                "joint_latent"
                            ]
                        )
                    )

                    tc_logits = (
                        tc_outputs["real"]
                    )

                    tc_logits_permuted = (
                        tc_outputs["permuted"]
                    )

                loss_dict = loss_fn(

                    outputs,

                    batch,

                    tc_logits=
                    tc_logits,

                    tc_logits_permuted=
                    tc_logits_permuted
                )

            metrics_tracker.update(
                loss_dict
            )

            last_latent_report = (

                latent_monitor.monitor(

                    epoch=epoch,

                    latents=
                    outputs["latents"]
                )
            )

    return {

        "metrics":

            metrics_tracker
            .averages(),

        "latent_report":

            last_latent_report
    }