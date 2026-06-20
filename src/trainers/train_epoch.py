# src/trainers/train_epoch.py

import torch

from torch.cuda.amp import (
    autocast
)


def train_epoch(

    model,

    loader,

    optimizer,

    loss_fn,

    metrics_tracker,

    latent_monitor,

    factorvae_scheduler,

    device,

    cfg,

    epoch,

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

    disc_loss_history = []
    disc_real_acc_history = []
    disc_perm_acc_history = []

    for batch in loader:

        batch = {

            k: (
                v.to(device)

                if torch.is_tensor(v)

                else v
            )

            for k, v in batch.items()
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

        # --------------------------------------------------
        # FactorVAE Discriminator Update
        # --------------------------------------------------

        disc_stats = (

            factorvae_scheduler
            .discriminator_step(

                outputs[
                    "joint_latent"
                ]
            )
        )

        disc_loss_history.append(

            float(
                disc_stats[
                    "disc_loss"
                ]
            )
        )

        disc_real_acc_history.append(

            float(
                disc_stats[
                    "disc_real_acc"
                ]
            )
        )

        disc_perm_acc_history.append(

            float(
                disc_stats[
                    "disc_perm_acc"
                ]
            )
        )

        # --------------------------------------------------
        # Generator TC logits
        # --------------------------------------------------

        with autocast(

            enabled=amp_enabled
        ):

            tc_outputs = (

                factorvae_scheduler
                .generator_logits(

                    outputs[
                        "joint_latent"
                    ]
                )
            )

            outputs[
                "tc_logits"
            ] = tc_outputs[
                "real"
            ]

            outputs[
                "tc_logits_permuted"
            ] = tc_outputs[
                "permuted"
            ]

            loss_dict = loss_fn(

                outputs,

                batch
            )

            loss = loss_dict[
                "total"
            ]

        # --------------------------------------------------
        # Generator Update
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------

        loss_dict[
            "disc_loss"
        ] = sum(
            disc_loss_history[-1:]
        )

        loss_dict[
            "disc_real_acc"
        ] = sum(
            disc_real_acc_history[-1:]
        )

        loss_dict[
            "disc_perm_acc"
        ] = sum(
            disc_perm_acc_history[-1:]
        )

        metrics_tracker.update(
            loss_dict
        )

        last_latent_report = (

            latent_monitor.monitor(

                epoch=epoch,

                latents=
                outputs[
                    "latents"
                ]
            )
        )

    metrics = (
        metrics_tracker
        .averages()
    )

    if len(disc_loss_history) > 0:

        metrics[
            "disc_loss"
        ] = sum(
            disc_loss_history
        ) / len(
            disc_loss_history
        )

        metrics[
            "disc_real_acc"
        ] = sum(
            disc_real_acc_history
        ) / len(
            disc_real_acc_history
        )

        metrics[
            "disc_perm_acc"
        ] = sum(
            disc_perm_acc_history
        ) / len(
            disc_perm_acc_history
        )

    return {

        "metrics":
            metrics,

        "latent_report":
            last_latent_report
    }