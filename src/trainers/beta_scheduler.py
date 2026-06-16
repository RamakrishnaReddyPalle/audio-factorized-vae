# src/trainers/beta_scheduler.py

from pathlib import Path
import math

from src.utils.config_loader import (
    load_yaml
)


class BetaScheduler:

    def __init__(
        self,
        project_root
    ):

        train_cfg = load_yaml(

            Path(project_root)
            /
            "configs"
            /
            "train_config.yaml"
        )

        model_cfg = load_yaml(

            Path(project_root)
            /
            "configs"
            /
            "model_config.yaml"
        )

        scheduler_cfg = (

            train_cfg[
                "beta_scheduler"
            ]
        )

        self.enabled = (
            scheduler_cfg.get(
                "enabled",
                False
            )
        )

        self.schedule_type = (
            scheduler_cfg.get(
                "type",
                "linear"
            )
        )

        self.warmup_epochs = (
            scheduler_cfg.get(
                "warmup_epochs",
                30
            )
        )

        self.cycle_length = (
            scheduler_cfg.get(
                "cycle_length",
                40
            )
        )

        self.min_scale = (
            scheduler_cfg.get(
                "min_scale",
                0.05
            )
        )

        self.cycle_factors = (
            scheduler_cfg.get(

                "cyclical_factors",

                [
                    "excitation",
                    "fidelity"
                ]
            )
        )

        self.final_betas = dict(

            model_cfg[
                "losses"
            ][
                "kl"
            ]
        )

    def linear_warmup(
        self,
        epoch
    ):

        return min(

            1.0,

            epoch
            /
            max(
                1,
                self.warmup_epochs
            )
        )

    def cyclical_scale(
        self,
        epoch
    ):

        phase = (

            epoch
            %
            self.cycle_length

        ) / self.cycle_length

        cosine = (

            0.5
            *
            (
                1.0
                -
                math.cos(
                    2.0
                    *
                    math.pi
                    *
                    phase
                )
            )
        )

        return (

            self.min_scale
            +

            (
                1.0
                -
                self.min_scale
            )
            *
            cosine
        )

    def get_beta_dict(
        self,
        epoch
    ):

        if not self.enabled:

            return dict(
                self.final_betas
            )

        warmup_scale = (
            self.linear_warmup(
                epoch
            )
        )

        cycle_scale = (
            self.cyclical_scale(
                epoch
            )
        )

        beta_dict = {}

        for name, value in (

            self.final_betas.items()
        ):

            scaled_value = (

                value
                *
                warmup_scale
            )

            if (

                self.schedule_type
                ==
                "cyclical"

                and

                name
                in
                self.cycle_factors
            ):

                scaled_value = (

                    scaled_value
                    *
                    cycle_scale
                )

            beta_dict[name] = (
                scaled_value
            )

        return beta_dict