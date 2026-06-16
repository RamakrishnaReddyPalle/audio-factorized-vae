# src/dataset/feature_loader.py

from pathlib import Path

from torch.utils.data import DataLoader

from src.dataset.feature_dataset import (
    FeatureDataset
)

from src.dataset.feature_collator import (
    FeatureCollator
)

from src.dataset.bucket_sampler import (
    BucketBatchSampler
)

from src.utils.config_loader import (
    load_yaml
)


def build_dataloader(
    project_root,
    split
):

    data_cfg = load_yaml(

        Path(project_root)
        /
        "configs"
        /
        "data_config.yaml"
    )

    train_cfg = load_yaml(

        Path(project_root)
        /
        "configs"
        /
        "train_config.yaml"
    )

    inventory_csv = (

        Path(project_root)

        /
        data_cfg["dataset"]
        ["metadata_dir"]

        /
        "feature_inventory_v2.csv"
    )

    dataset = FeatureDataset(

        inventory_csv=
            inventory_csv,

        split=
            split
    )

    batch_size = (

        train_cfg[
            "training"
        ][
            "batch_size"
        ]
    )

    if split == "train":

        batch_sampler = (

            BucketBatchSampler(

                dataset,

                batch_size=
                    batch_size,

                drop_last=
                    train_cfg[
                        "training"
                    ][
                        "drop_last"
                    ]
            )
        )

        loader = DataLoader(

            dataset,

            batch_sampler=
                batch_sampler,

            num_workers=
                train_cfg[
                    "training"
                ][
                    "num_workers"
                ],

            pin_memory=
                train_cfg[
                    "training"
                ][
                    "pin_memory"
                ],

            persistent_workers=
                train_cfg[
                    "training"
                ][
                    "persistent_workers"
                ],

            prefetch_factor=
                train_cfg[
                    "training"
                ][
                    "prefetch_factor"
                ],

            collate_fn=
                FeatureCollator()
        )

    else:

        loader = DataLoader(

            dataset,

            batch_size=
                batch_size,

            shuffle=False,

            num_workers=
                train_cfg[
                    "training"
                ][
                    "num_workers"
                ],

            pin_memory=
                train_cfg[
                    "training"
                ][
                    "pin_memory"
                ],

            persistent_workers=
                train_cfg[
                    "training"
                ][
                    "persistent_workers"
                ],

            prefetch_factor=
                train_cfg[
                    "training"
                ][
                    "prefetch_factor"
                ],

            collate_fn=
                FeatureCollator()
        )

    return loader