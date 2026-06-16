# src/preprocessing/feature_extractors/instantaneous_frequency.py

import numpy as np


class InstantaneousFrequencyExtractor:

    def __init__(
        self,
        hop_length,
        sample_rate
    ):

        self.hop_length = hop_length
        self.sample_rate = sample_rate

    def extract(
        self,
        phase
    ):

        phase_unwrapped = np.unwrap(
            phase,
            axis=1
        )

        delta_phase = np.diff(
            phase_unwrapped,
            axis=1
        )

        if_feature = (

            self.sample_rate

            /

            (
                2
                * np.pi
                * self.hop_length
            )

        ) * delta_phase

        if_feature = np.pad(

            if_feature,

            (
                (0, 0),
                (1, 0)
            ),

            mode="edge"
        )

        median = np.median(
            if_feature
        )

        mad = np.median(
            np.abs(
                if_feature - median
            )
        ) + 1e-8

        if_feature = (
            if_feature - median
        ) / mad

        return if_feature.astype(
            np.float32
        )