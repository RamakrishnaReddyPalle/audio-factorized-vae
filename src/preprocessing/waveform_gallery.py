# src/preprocessing/waveform_gallery.py

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt


def create_waveform_gallery(
        filepaths,
        save_path,
        max_files=16
):

    filepaths = filepaths[:max_files]

    rows = len(filepaths)

    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(12, rows * 2)
    )

    if rows == 1:
        axes = [axes]

    for ax, filepath in zip(axes, filepaths):

        y, sr = librosa.load(
            filepath,
            sr=None,
            mono=True
        )

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax
        )

        ax.set_title(
            Path(filepath).name,
            fontsize=8
        )

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()