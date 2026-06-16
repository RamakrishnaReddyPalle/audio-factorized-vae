# src/trainers/checkpoint_manager.py

from pathlib import Path

import torch

from src.utils.config_loader import (
    load_yaml
)


class CheckpointManager:

    def __init__(
        self,
        project_root
    ):

        self.project_root = Path(
            project_root
        )

        data_cfg = load_yaml(

            self.project_root
            /
            "configs"
            /
            "data_config.yaml"
        )

        train_cfg = load_yaml(

            self.project_root
            /
            "configs"
            /
            "train_config.yaml"
        )

        self.save_every = (

            train_cfg[
                "checkpoint"
            ][
                "save_every"
            ]
        )

        self.keep_last = (

            train_cfg[
                "checkpoint"
            ][
                "keep_last"
            ]
        )

        self.checkpoint_dir = (

            self.project_root
            /
            data_cfg[
                "outputs"
            ][
                "checkpoint_dir"
            ]
        )

        self.checkpoint_dir.mkdir(

            parents=True,
            exist_ok=True
        )

        self.best_metric = None

    def checkpoint_state(

        self,

        epoch,

        model,

        optimizer,

        scheduler,

        metrics
    ):

        state = {

            "epoch":
                epoch,

            "model":
                model.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "metrics":
                metrics
        }

        if scheduler is not None:

            state[
                "scheduler"
            ] = scheduler.state_dict()

        return state

    def save_latest(

        self,

        epoch,

        model,

        optimizer,

        scheduler,

        metrics
    ):

        state = self.checkpoint_state(

            epoch,

            model,

            optimizer,

            scheduler,

            metrics
        )

        path = (

            self.checkpoint_dir
            /
            "latest.pt"
        )

        torch.save(
            state,
            path
        )

    def save_epoch(

        self,

        epoch,

        model,

        optimizer,

        scheduler,

        metrics
    ):

        if (

            epoch %
            self.save_every

            != 0
        ):

            return

        state = self.checkpoint_state(

            epoch,

            model,

            optimizer,

            scheduler,

            metrics
        )

        path = (

            self.checkpoint_dir
            /
            f"epoch_{epoch:04d}.pt"
        )

        torch.save(
            state,
            path
        )

        self.cleanup_old()

    def save_best(

        self,

        metric_value,

        epoch,

        model,

        optimizer,

        scheduler,

        metrics
    ):

        if (

            self.best_metric
            is None
        ):

            self.best_metric = (
                metric_value
            )

        if (

            metric_value
            <
            self.best_metric
        ):

            self.best_metric = (
                metric_value
            )

            state = (

                self.checkpoint_state(

                    epoch,

                    model,

                    optimizer,

                    scheduler,

                    metrics
                )
            )

            path = (

                self.checkpoint_dir
                /
                "best.pt"
            )

            torch.save(
                state,
                path
            )

    def cleanup_old(
        self
    ):

        checkpoints = sorted(

            self.checkpoint_dir.glob(
                "epoch_*.pt"
            )
        )

        if (

            len(checkpoints)

            <=

            self.keep_last
        ):

            return

        remove_count = (

            len(checkpoints)

            -
            self.keep_last
        )

        for ckpt in checkpoints[
            :remove_count
        ]:

            ckpt.unlink()

    def load(
        self,
        checkpoint_path
    ):

        checkpoint_path = Path(
            checkpoint_path
        )

        return torch.load(

            checkpoint_path,

            map_location="cpu"
        )

    def resume(

        self,

        checkpoint_path,

        model,

        optimizer=None,

        scheduler=None
    ):

        state = self.load(
            checkpoint_path
        )

        model.load_state_dict(

            state["model"]
        )

        if (

            optimizer is not None

            and

            "optimizer"
            in state
        ):

            optimizer.load_state_dict(

                state[
                    "optimizer"
                ]
            )

        if (

            scheduler is not None

            and

            "scheduler"
            in state
        ):

            scheduler.load_state_dict(

                state[
                    "scheduler"
                ]
            )

        return state