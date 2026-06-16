# src/preprocessing/plot_dataset_statistics.py

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DatasetVisualizer:

    def __init__(self, output_dir):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
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

        plt.title("Duration Distribution")

        plt.tight_layout()

        plt.savefig(
            self.output_dir /
            "duration_distribution.png"
        )

        plt.close()

    def rms_distribution(self, df):

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="condition",
            y="rms"
        )

        plt.title("RMS Distribution")

        plt.tight_layout()

        plt.savefig(
            self.output_dir /
            "rms_distribution.png"
        )

        plt.close()

    def sample_rate_distribution(self, df):

        plt.figure(figsize=(8, 5))

        sns.countplot(
            data=df,
            x="sample_rate"
        )

        plt.title("Sample Rate Distribution")

        plt.tight_layout()

        plt.savefig(
            self.output_dir /
            "sample_rate_distribution.png"
        )

        plt.close()

    def condition_counts(self, df):

        plt.figure(figsize=(6, 5))

        sns.countplot(
            data=df,
            x="condition"
        )

        plt.title("Condition Counts")

        plt.tight_layout()

        plt.savefig(
            self.output_dir /
            "condition_counts.png"
        )

        plt.close()

    def speaker_counts(self, df):

        plt.figure(figsize=(6, 5))

        sns.countplot(
            data=df,
            x="speaker_folder"
        )

        plt.title("Speaker Counts")

        plt.tight_layout()

        plt.savefig(
            self.output_dir /
            "speaker_counts.png"
        )

        plt.close()