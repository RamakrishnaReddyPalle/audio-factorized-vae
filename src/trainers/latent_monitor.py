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

        z = z.detach()

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

        mean_value = (
            z.mean().item()
        )

        std_value = (
            z.std().item()
        )

        avg_variance = (
            variance.mean().item()
        )

        collapse_ratio = float(
            dead.float().mean()
        )

        collapsed = (

            avg_variance
            <
            self.collapse_threshold
        )

        return {

            "mean":
                mean_value,

            "std":
                std_value,

            "avg_variance":
                avg_variance,

            "active_dims":
                int(
                    active.sum()
                ),

            "dead_dims":
                int(
                    dead.sum()
                ),

            "collapse_ratio":
                collapse_ratio,

            "collapsed":
                collapsed,

            "min_variance":
                variance.min().item(),

            "max_variance":
                variance.max().item()
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

        report = {

            "epoch":
                epoch
        }

        for name, data in (

            latents.items()
        ):

            if isinstance(
                data,
                dict
            ):

                if "z" in data:

                    z = data["z"]

                else:

                    continue

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
            f"LATENT MONITOR | Epoch {report.get('epoch', '?')}"
        )

        print(
            "=" * 60
        )

        for name, stats in (

            report.items()
        ):

            if name == "epoch":

                continue

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

                f"min_var={stats['min_variance']:.6f}"
            )

            print(

                f"max_var={stats['max_variance']:.6f}"
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

            print(

                f"collapsed={stats['collapsed']}"
            )