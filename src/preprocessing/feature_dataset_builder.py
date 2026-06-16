# src/preprocessing/feature_dataset_builder.py

from pathlib import Path
import json

import librosa
import numpy as np

from src.preprocessing.feature_extractors.feature_pipeline import (
    FeaturePipeline
)


class FeatureDatasetBuilder:

    def __init__(
        self,
        cfg
    ):

        self.pipeline = (
            FeaturePipeline(
                cfg
            )
        )

    def save_feature(
        self,
        output_dir,
        feature_name,
        feature_array
    ):

        np.save(

            output_dir
            / f"{feature_name}.npy",

            feature_array.astype(
                np.float32
            )
        )

    def process_fragment(
        self,
        fragment_path,
        output_dir,
        metadata
    ):

        output_dir.mkdir(

            parents=True,

            exist_ok=True
        )

        waveform, sr = librosa.load(

            fragment_path,

            sr=None,

            mono=True
        )

        features = (
            self.pipeline.extract(
                waveform
            )
        )

        # --------------------------------------------------
        # Save Every Feature Automatically
        # --------------------------------------------------

        shape_metadata = {}

        for feature_name, feature_array in features.items():

            # stft is complex
            if feature_name == "stft":
                continue

            self.save_feature(

                output_dir,

                feature_name,

                feature_array
            )

            if hasattr(
                feature_array,
                "shape"
            ):

                shape_metadata[
                    f"{feature_name}_shape"
                ] = list(
                    feature_array.shape
                )

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        feature_meta = {

            **metadata,

            **shape_metadata,

            "available_features":

                sorted(
                    [
                        k
                        for k in features.keys()
                        if k != "stft"
                    ]
                ),

            "sample_rate":
                sr
        }

        with open(

            output_dir
            / "feature_metadata.json",

            "w"

        ) as f:

            json.dump(

                feature_meta,

                f,

                indent=4
            )

        return feature_meta