# src/preprocessing/feature_extractors/magnitude_phase.py

import librosa
import numpy as np


class MagnitudePhaseExtractor:

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        window,
        eps=1e-8
    ):

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.eps = eps

    def compute_stft(
        self,
        waveform
    ):

        return librosa.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window
        )

    def extract(
        self,
        waveform
    ):

        stft = self.compute_stft(
            waveform
        )

        magnitude = np.abs(
            stft
        )

        magnitude_db = librosa.amplitude_to_db(
            magnitude + self.eps,
            ref=np.max
        )

        magnitude_db = (
            magnitude_db + 80.0
        ) / 80.0

        phase = np.angle(
            stft
        )

        phase_sin = np.sin(
            phase
        )

        phase_cos = np.cos(
            phase
        )

        return {

            "stft":
                stft,

            "magnitude":
                magnitude_db.astype(
                    np.float32
                ),

            "phase":
                phase.astype(
                    np.float32
                ),

            "phase_sin":
                phase_sin.astype(
                    np.float32
                ),

            "phase_cos":
                phase_cos.astype(
                    np.float32
                )
        }