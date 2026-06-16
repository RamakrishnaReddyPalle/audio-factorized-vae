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

    device,

    cfg
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

                for k,v in batch.items()
            }

            with autocast(

                enabled=amp_enabled
            ):

                outputs = model(
                    batch
                )

                loss_dict = loss_fn(

                    outputs,

                    batch
                )

            metrics_tracker.update(
                loss_dict
            )

            last_latent_report = (

                latent_monitor.monitor(

                    epoch=0,

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