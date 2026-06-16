# src/trainers/trainer.py

from pathlib import Path

import torch

from torch.cuda.amp import (
    GradScaler
)

from src.trainers.train_epoch import (
    train_epoch
)

from src.trainers.validate_epoch import (
    validate_epoch
)

from src.trainers.metrics_tracker import (
    MetricsTracker
)

from src.trainers.beta_scheduler import (
    BetaScheduler
)

from src.trainers.checkpoint_manager import (
    CheckpointManager
)

from src.trainers.latent_monitor import (
    LatentMonitor
)

from src.trainers.memory_monitor import (
    MemoryMonitor
)


class Trainer:

    def __init__(

        self,

        model,

        train_loader,

        val_loader,

        optimizer,

        scheduler,

        loss_fn,

        cfg,

        project_root,

        device
    ):

        self.model = model

        self.train_loader = (
            train_loader
        )

        self.val_loader = (
            val_loader
        )

        self.optimizer = (
            optimizer
        )

        self.scheduler = (
            scheduler
        )

        self.loss_fn = (
            loss_fn
        )

        self.cfg = cfg

        self.device = (
            device
        )

        self.project_root = (
            project_root
        )

        self.beta_scheduler = (

            BetaScheduler(
                project_root
            )
        )

        self.checkpoints = (

            CheckpointManager(
                project_root
            )
        )

        self.latent_monitor = (

            LatentMonitor(
                project_root
            )
        )

        self.memory_monitor = (
            MemoryMonitor()
        )

        self.train_metrics = (
            MetricsTracker()
        )

        self.val_metrics = (
            MetricsTracker()
        )

        self.scaler = (

            GradScaler()

            if

            cfg[
                "mixed_precision"
            ][
                "enabled"
            ]

            else None
        )

        self.best_loss = None

        self.patience_counter = 0

    def fit(
        self
    ):

        epochs = (

            self.cfg[
                "training"
            ][
                "epochs"
            ]
        )

        early_cfg = (

            self.cfg[
                "early_stopping"
            ]
        )

        for epoch in range(

            1,

            epochs + 1
        ):

            print()

            print(
                "=" * 80
            )

            print(
                f"Epoch {epoch}"
            )

            print(
                "=" * 80
            )

            # --------------------------------------------------
            # Beta Warmup
            # --------------------------------------------------

            beta_dict = (

                self.beta_scheduler
                .get_beta_dict(
                    epoch
                )
            )

            self.loss_fn.current_epoch = epoch

            self.loss_fn.total_epochs = epochs

            self.loss_fn.cfg[
                "losses"
            ][
                "kl"
            ] = beta_dict

            # --------------------------------------------------
            # Memory Tracking
            # --------------------------------------------------

            self.memory_monitor.start_epoch()

            train_result = (

                train_epoch(

                    model=
                        self.model,

                    loader=
                        self.train_loader,

                    optimizer=
                        self.optimizer,

                    loss_fn=
                        self.loss_fn,

                    metrics_tracker=
                        self.train_metrics,

                    latent_monitor=
                        self.latent_monitor,

                    device=
                        self.device,

                    cfg=
                        self.cfg,

                    scaler=
                        self.scaler
                )
            )

            val_result = (

                validate_epoch(

                    model=
                        self.model,

                    loader=
                        self.val_loader,

                    loss_fn=
                        self.loss_fn,

                    metrics_tracker=
                        self.val_metrics,

                    latent_monitor=
                        self.latent_monitor,

                    device=
                        self.device,

                    cfg=
                        self.cfg
                )
            )

            memory_stats = (
                self.memory_monitor.end_epoch()
            )

            train_loss = (

                train_result[
                    "metrics"
                ][
                    "total"
                ]
            )

            val_loss = (

                val_result[
                    "metrics"
                ][
                    "total"
                ]
            )

            print()

            print(
                f"Train Loss: {train_loss:.4f}"
            )

            print(
                f"Val Loss:   {val_loss:.4f}"
            )

            print()

            print(
                memory_stats
            )

            self.latent_monitor.print_report(

                train_result[
                    "latent_report"
                ]
            )

            if self.scheduler is not None:

                self.scheduler.step()

            self.checkpoints.save_latest(

                epoch,

                self.model,

                self.optimizer,

                self.scheduler,

                val_result[
                    "metrics"
                ]
            )

            self.checkpoints.save_epoch(

                epoch,

                self.model,

                self.optimizer,

                self.scheduler,

                val_result[
                    "metrics"
                ]
            )

            self.checkpoints.save_best(

                val_loss,

                epoch,

                self.model,

                self.optimizer,

                self.scheduler,

                val_result[
                    "metrics"
                ]
            )

            if (

                self.best_loss
                is None
            ):

                self.best_loss = (
                    val_loss
                )

            elif (

                val_loss
                <
                self.best_loss
            ):

                self.best_loss = (
                    val_loss
                )

                self.patience_counter = 0

            else:

                self.patience_counter += 1

            if (

                early_cfg[
                    "enabled"
                ]

                and

                self.patience_counter

                >=

                early_cfg[
                    "patience"
                ]
            ):

                print()

                print(
                    "Early stopping triggered."
                )

                break

        print()

        print(
            "Training Finished."
        )