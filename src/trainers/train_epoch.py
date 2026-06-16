# src/trainers/train_epoch.py

import torch

from torch.cuda.amp import (
    autocast,
    GradScaler
)


def train_epoch(

    model,

    loader,

    optimizer,

    loss_fn,

    metrics_tracker,

    latent_monitor,

    device,

    cfg,

    scaler=None
):

    model.train()

    metrics_tracker.reset()

    last_latent_report = {}

    clip_norm = (

        cfg["gradient"]
        ["clip_norm"]
    )

    amp_enabled = (

        cfg["mixed_precision"]
        ["enabled"]
    )

    for batch in loader:

        batch = {

            k: (
                v.to(device)

                if torch.is_tensor(v)

                else v
            )

            for k,v in batch.items()
        }

        optimizer.zero_grad(
            set_to_none=True
        )

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

            loss = loss_dict[
                "total"
            ]

        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                clip_norm
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                clip_norm
            )

            optimizer.step()

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