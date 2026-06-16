# src/dataset/feature_collator.py

import torch
import torch.nn.functional as F


class FeatureCollator:

    FEATURE_NAMES = [

        "logmel",

        "mr_mag_256",
        "mr_mag_512",
        "mr_mag_1024",

        "magnitude",

        "if",

        "modgd",

        "phase_sin",
        "phase_cos"
    ]

    def pad_feature(
        self,
        tensors
    ):

        max_time = max(

            x.shape[-1]

            for x in tensors
        )

        padded = []

        for x in tensors:

            pad_amount = (

                max_time
                -
                x.shape[-1]
            )

            padded.append(

                F.pad(

                    x,

                    (
                        0,
                        pad_amount
                    )
                )
            )

        return torch.stack(
            padded
        )

    def __call__(
        self,
        batch
    ):

        output = {}

        for feature_name in self.FEATURE_NAMES:

            tensors = [

                sample[
                    feature_name
                ]

                for sample in batch
            ]

            output[
                feature_name
            ] = self.pad_feature(
                tensors
            )

        output["lengths"] = torch.tensor(

            [
                sample["length"]

                for sample
                in batch
            ]
        )

        output["speaker"] = [

            x["speaker"]

            for x in batch
        ]

        output["condition"] = [

            x["condition"]

            for x in batch
        ]

        output["split"] = [

            x["split"]

            for x in batch
        ]

        return output