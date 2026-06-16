# src/trainers/latent_monitor.py

from pathlib import Path

import torch

from src.utils.config_loader import (
    load_yaml
)


class LatentMonitor:

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

        monitor_cfg = (

            train_cfg[
                "latent_monitor"
            ]
        )

        self.enabled = (

            monitor_cfg[
                "enabled"
            ]
        )

        self.monitor_every = (

            monitor_cfg[
                "monitor_every"
            ]
        )

        self.collapse_threshold = (

            monitor_cfg[
                "collapse_threshold"
            ]
        )

        self.variance_threshold = (

            monitor_cfg[
                "variance_threshold"
            ]
        )

    def analyze_latent(

        self,

        z
    ):

        variance = torch.var(

            z,

            dim=0,

            unbiased=False
        )

        active = (

            variance
            >
            self.variance_threshold
        )

        dead = (

            variance
            <=
            self.variance_threshold
        )

        return {

            "mean":

                z.mean().item(),

            "std":

                z.std().item(),

            "avg_variance":

                variance.mean().item(),

            "active_dims":

                int(
                    active.sum()
                ),

            "dead_dims":

                int(
                    dead.sum()
                ),

            "collapse_ratio":

                float(
                    dead.float()
                    .mean()
                )
        }

    def monitor(

        self,

        epoch,

        latents
    ):

        if not self.enabled:

            return {}

        if (

            epoch
            %
            self.monitor_every

            != 0
        ):

            return {}

        report = {}

        for name,data in (

            latents.items()
        ):

            if isinstance(
                data,
                dict
            ):

                z = data["z"]

            else:

                z = data

            report[name] = (

                self.analyze_latent(
                    z
                )
            )

        return report

    def print_report(

        self,

        report
    ):

        if len(report) == 0:

            return

        print()

        print(
            "=" * 60
        )

        print(
            "LATENT MONITOR"
        )

        print(
            "=" * 60
        )

        for name,stats in (

            report.items()
        ):

            print()

            print(
                name.upper()
            )

            print(

                f"mean={stats['mean']:.4f}"
            )

            print(

                f"std={stats['std']:.4f}"
            )

            print(

                f"variance={stats['avg_variance']:.6f}"
            )

            print(

                f"active={stats['active_dims']}"
            )

            print(

                f"dead={stats['dead_dims']}"
            )

            print(

                f"collapse={stats['collapse_ratio']:.3f}"
            )