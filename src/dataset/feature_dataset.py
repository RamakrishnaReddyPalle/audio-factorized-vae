# src/dataset/feature_dataset.py

from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):

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

    def __init__(
        self,
        inventory_csv,
        split=None
    ):

        self.inventory_path = Path(
            inventory_csv
        )

        self.inventory = pd.read_csv(
            self.inventory_path
        )

        if split is not None:

            self.inventory = (

                self.inventory[
                    self.inventory["split"]
                    == split
                ]
                .reset_index(drop=True)
            )

        self.feature_root = (

            self.inventory_path.parent.parent
            /
            "features"
        )

    def __len__(self):

        return len(
            self.inventory
        )

    def __getitem__(
        self,
        idx
    ):

        row = self.inventory.iloc[
            idx
        ]

        speaker = row["speaker"]

        source_file = Path(
            row["source_file"]
        ).stem

        fragment_id = int(
            row["fragment_id"]
        )

        feature_dir = (

            self.feature_root
            /
            speaker
            /
            source_file
            /
            f"fragment_{fragment_id:03d}"
        )

        sample = {}

        for feature_name in self.FEATURE_NAMES:

            x = np.load(

                feature_dir
                /
                f"{feature_name}.npy"
            )

            sample[
                feature_name
            ] = torch.tensor(
                x,
                dtype=torch.float32
            )

        sample["length"] = (

            sample["logmel"]
            .shape[-1]
        )

        sample["speaker"] = (
            speaker
        )

        sample["condition"] = (
            row["condition"]
        )

        sample["split"] = (
            row["split"]
        )

        sample["fragment_id"] = (
            fragment_id
        )

        sample["source_file"] = (
            source_file
        )

        sample["relative_position"] = float(
            row["relative_position"]
        )

        return sample