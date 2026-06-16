# src/visualization/audit_plots.py

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import librosa.display
import numpy as np


class AuditPlots:

    def __init__(self, save_dir):

        self.save_dir = Path(save_dir)

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def duration_histogram(self, df):

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x="duration_sec",
            hue="condition",
            bins=20
        )

        plt.tight_layout()

        plt.savefig(
            self.save_dir /
            "duration_hist.png"
        )

        plt.close()

    def rms_histogram(self, df):

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x="rms",
            hue="condition"
        )

        plt.tight_layout()

        plt.savefig(
            self.save_dir /
            "rms_hist.png"
        )

        plt.close()

    def speaker_condition_matrix(self, df):

        table = df.groupby(
            ["speaker", "condition"]
        ).size().unstack(fill_value=0)

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            table,
            annot=True,
            fmt="d"
        )

        plt.tight_layout()

        plt.savefig(
            self.save_dir /
            "speaker_condition_matrix.png"
        )

        plt.close()

    def waveform_gallery(
        self,
        filepaths,
        max_files=12
    ):

        filepaths = filepaths[:max_files]

        fig, axes = plt.subplots(
            len(filepaths),
            1,
            figsize=(12, 2 * len(filepaths))
        )

        if len(filepaths) == 1:
            axes = [axes]

        for ax, fp in zip(axes, filepaths):

            y, sr = librosa.load(
                fp,
                sr=None,
                mono=True
            )

            librosa.display.waveshow(
                y,
                sr=sr,
                ax=ax
            )

            ax.set_title(
                Path(fp).name,
                fontsize=8
            )

        plt.tight_layout()

        plt.savefig(
            self.save_dir /
            "waveform_gallery.png"
        )

        plt.close()

    def spectrogram_gallery(
        self,
        filepaths,
        max_files=12
    ):

        filepaths = filepaths[:max_files]

        fig, axes = plt.subplots(
            len(filepaths),
            1,
            figsize=(12, 2.5 * len(filepaths))
        )

        if len(filepaths) == 1:
            axes = [axes]

        for ax, fp in zip(axes, filepaths):

            y, sr = librosa.load(
                fp,
                sr=None,
                mono=True
            )

            stft = librosa.stft(y)

            db = librosa.amplitude_to_db(
                np.abs(stft),
                ref=np.max
            )

            librosa.display.specshow(
                db,
                sr=sr,
                x_axis="time",
                y_axis="hz",
                ax=ax
            )

            ax.set_title(
                Path(fp).name,
                fontsize=8
            )

        plt.tight_layout()

        plt.savefig(
            self.save_dir /
            "spectrogram_gallery.png"
        )

        plt.close()