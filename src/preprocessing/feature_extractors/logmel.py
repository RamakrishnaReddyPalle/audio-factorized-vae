# src/preprocessing/feature_extractors/logmel.py

import librosa
import numpy as np


class LogMelExtractor:

    def __init__(
        self,
        sample_rate,
        n_fft,
        hop_length,
        win_length,
        n_mels,
        fmin,
        fmax,
        eps=1e-8
    ):

        self.sample_rate = sample_rate

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.n_mels = n_mels

        self.fmin = fmin
        self.fmax = fmax

        self.eps = eps

    def extract(
        self,
        waveform
    ):

        mel = librosa.feature.melspectrogram(

            y=waveform,

            sr=self.sample_rate,

            n_fft=self.n_fft,

            hop_length=self.hop_length,

            win_length=self.win_length,

            n_mels=self.n_mels,

            fmin=self.fmin,

            fmax=self.fmax,

            power=2.0
        )

        mel_db = librosa.power_to_db(

            mel + self.eps,

            ref=np.max
        )

        mel_db = (
            mel_db + 80.0
        ) / 80.0

        return mel_db.astype(
            np.float32
        )