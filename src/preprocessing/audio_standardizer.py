# src/preprocessing/audio_standardizer.py

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.utils.audio_loader import AudioLoader


class AudioStandardizer:

    def __init__(
        self,
        target_sr=16000,
        mono=True,
        peak_normalize=True
    ):

        self.target_sr = target_sr
        self.mono = mono
        self.peak_normalize = peak_normalize

    def process_file(
        self,
        input_path,
        output_path
    ):

        y, sr = AudioLoader.load(input_path)

        original_sr = sr

        original_channels = 1

        if y.ndim > 1:
            original_channels = y.shape[0]

        if self.mono:
            y = AudioLoader.to_mono(y)

        y = librosa.resample(
            y,
            orig_sr=sr,
            target_sr=self.target_sr
        )

        rms_before = AudioLoader.rms(y)

        peak_before = AudioLoader.peak(y)

        if self.peak_normalize:

            if peak_before > 0:
                y = y / peak_before

        rms_after = AudioLoader.rms(y)

        peak_after = AudioLoader.peak(y)

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        sf.write(
            output_path,
            y.astype(np.float32),
            self.target_sr
        )

        return {

            "original_path": str(input_path),

            "processed_path": str(output_path),

            "original_sr": original_sr,

            "processed_sr": self.target_sr,

            "original_channels":
                original_channels,

            "processed_channels": 1,

            "duration_sec":
                len(y) / self.target_sr,

            "rms_before":
                rms_before,

            "rms_after":
                rms_after,

            "peak_before":
                peak_before,

            "peak_after":
                peak_after
        }