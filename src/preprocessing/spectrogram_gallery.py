# src/preprocessing/spectrogram_gallery.py

from pathlib import Path

import librosa
import librosa.display

import matplotlib.pyplot as plt
import numpy as np


def create_spectrogram_gallery(
        filepaths,
        save_path,
        max_files=16
):

    filepaths = filepaths[:max_files]

    rows = len(filepaths)

    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(12, rows * 2.5)
    )

    if rows == 1:
        axes = [axes]

    for ax, filepath in zip(axes, filepaths):

        y, sr = librosa.load(
            filepath,
            sr=None,
            mono=True
        )

        spec = librosa.stft(y)

        spec_db = librosa.amplitude_to_db(
            np.abs(spec),
            ref=np.max
        )

        librosa.display.specshow(
            spec_db,
            sr=sr,
            x_axis="time",
            y_axis="hz",
            ax=ax
        )

        ax.set_title(
            Path(filepath).name,
            fontsize=8
        )

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()