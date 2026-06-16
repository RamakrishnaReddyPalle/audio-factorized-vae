# src/preprocessing/feature_extractors/modgd.py

import numpy as np


class MODGDExtractor:

    def __init__(
        self,
        alpha=0.4,
        gamma=0.9,
        eps=1e-8
    ):

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def extract(
        self,
        stft
    ):

        phase = np.angle(
            stft
        )

        gd = np.gradient(
            phase,
            axis=0
        )

        magnitude = np.abs(
            stft
        )

        modgd = (

            np.sign(gd)

            *

            (
                np.abs(gd)
                ** self.alpha
            )

            /

            (
                magnitude
                ** self.gamma
                + self.eps
            )
        )

        modgd = np.tanh(
            modgd
        )

        return modgd.astype(
            np.float32
        )