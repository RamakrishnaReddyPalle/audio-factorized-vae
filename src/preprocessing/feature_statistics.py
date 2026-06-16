# src/preprocessing/feature_statistics.py

from pathlib import Path
import json

import numpy as np
from tqdm import tqdm


class FeatureStatisticsBuilder:

    def __init__(self):

        self.feature_names = [

            "magnitude",
            "logmel",
            "if",
            "modgd",

            "mr_mag_256",
            "mr_mag_512",
            "mr_mag_1024",

            "phase_sin",
            "phase_cos"
        ]

    def compute_stats(
        self,
        feature_inventory,
        feature_dir
    ):

        stats = {}

        for feature_name in self.feature_names:

            print(
                f"\nProcessing {feature_name}"
            )

            values = []

            for _, row in tqdm(

                feature_inventory.iterrows(),

                total=len(feature_inventory)
            ):

                speaker = row["speaker"]

                source_file = Path(
                    row["source_file"]
                ).stem

                fragment_id = int(
                    row["fragment_id"]
                )

                folder = (

                    feature_dir

                    / speaker

                    / source_file

                    / f"fragment_{fragment_id:03d}"
                )

                feature_path = (

                    folder
                    / f"{feature_name}.npy"
                )

                if not feature_path.exists():
                    continue

                x = np.load(
                    feature_path
                )

                values.append(
                    x.reshape(-1)
                )

            values = np.concatenate(
                values,
                axis=0
            )

            stats[feature_name] = {

                "mean":
                    float(
                        np.mean(values)
                    ),

                "std":
                    float(
                        np.std(values)
                    ),

                "min":
                    float(
                        np.min(values)
                    ),

                "max":
                    float(
                        np.max(values)
                    ),

                "p01":
                    float(
                        np.percentile(
                            values,
                            1
                        )
                    ),

                "p99":
                    float(
                        np.percentile(
                            values,
                            99
                        )
                    )
            }

        return stats

    def save(
        self,
        stats,
        output_path
    ):

        with open(
            output_path,
            "w"
        ) as f:

            json.dump(
                stats,
                f,
                indent=4
            )