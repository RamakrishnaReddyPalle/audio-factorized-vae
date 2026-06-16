# src/utils/audio_loader.py

import librosa
import numpy as np


class AudioLoader:

    @staticmethod
    def load(filepath):

        try:

            y, sr = librosa.load(
                str(filepath),
                sr=None,
                mono=False
            )

            return y, sr

        except Exception as e:

            raise RuntimeError(
                f"\nFailed loading:\n{filepath}\n\n{e}"
            )

    @staticmethod
    def to_mono(y):

        if y.ndim == 1:
            return y

        return librosa.to_mono(y)

    @staticmethod
    def rms(y):

        return float(
            np.sqrt(
                np.mean(y ** 2)
            )
        )

    @staticmethod
    def peak(y):

        return float(
            np.max(
                np.abs(y)
            )
        )

    @staticmethod
    def clipping_ratio(
        y,
        threshold=0.99
    ):

        return float(
            np.mean(
                np.abs(y) > threshold
            )
        )