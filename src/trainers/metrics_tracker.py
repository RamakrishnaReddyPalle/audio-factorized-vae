# src/trainers/metrics_tracker.py

import numpy as np


class MetricsTracker:

    def __init__(
        self
    ):

        self.history = {}

    def reset(
        self
    ):

        self.history = {}

    def update(
        self,
        loss_dict
    ):

        for key,value in (

            loss_dict.items()
        ):

            if hasattr(
                value,
                "item"
            ):

                value = value.item()

            if key not in self.history:

                self.history[key] = []

            self.history[key].append(
                float(value)
            )

    def averages(
        self
    ):

        output = {}

        for key,values in (

            self.history.items()
        ):

            output[key] = float(

                np.mean(values)
            )

        return output

    def summary_string(
        self
    ):

        avg = self.averages()

        pieces = []

        priority = [

            "total",

            "reconstruction",

            "multires",

            "orthogonality",

            "tc"
        ]

        for key in priority:

            if key in avg:

                pieces.append(

                    f"{key}: "
                    f"{avg[key]:.4f}"
                )

        return " | ".join(
            pieces
        )

    def state_dict(
        self
    ):

        return self.history

    def load_state_dict(
        self,
        state
    ):

        self.history = state