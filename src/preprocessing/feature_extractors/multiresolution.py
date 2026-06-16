# src/preprocessing/feature_extractors/multiresolution.py

import librosa
import numpy as np


class MultiResolutionExtractor:

    def __init__(
        self,
        sample_rate,
        fft_sizes,
        hop_lengths,
        win_lengths
    ):

        self.sample_rate = sample_rate

        self.fft_sizes = fft_sizes
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths

    def extract(
        self,
        waveform
    ):

        outputs = {}

        for n_fft, hop, win in zip(

            self.fft_sizes,

            self.hop_lengths,

            self.win_lengths
        ):

            stft = librosa.stft(

                waveform,

                n_fft=n_fft,

                hop_length=hop,

                win_length=win
            )

            magnitude = np.abs(
                stft
            )

            magnitude = librosa.amplitude_to_db(

                magnitude + 1e-8,

                ref=np.max
            )

            magnitude = (
                magnitude + 80.0
            ) / 80.0

            phase = np.angle(
                stft
            )

            outputs[f"mr_mag_{n_fft}"] = (
                magnitude.astype(
                    np.float32
                )
            )

            outputs[f"mr_phase_{n_fft}"] = (
                phase.astype(
                    np.float32
                )
            )

        return outputs