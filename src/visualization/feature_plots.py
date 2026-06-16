# src/visualization/feature_plots.py

import numpy as np
import matplotlib.pyplot as plt


def plot_feature(
    feature,
    title="Feature",
    figsize=(10, 5),
    cmap="viridis",
    show_colorbar=True
):

    plt.figure(
        figsize=figsize
    )

    plt.imshow(
        feature,
        aspect="auto",
        origin="lower",
        cmap=cmap
    )

    plt.title(title)

    plt.xlabel(
        "Frames"
    )

    plt.ylabel(
        "Frequency Bins"
    )

    if show_colorbar:
        plt.colorbar()

    plt.tight_layout()
    plt.show()


def plot_magnitude(
    magnitude
):

    plot_feature(
        magnitude,
        title="Magnitude Spectrogram"
    )


def plot_phase(
    phase
):

    plot_feature(
        phase,
        title="Phase Spectrogram",
        cmap="twilight"
    )


def plot_if(
    if_feature
):

    plot_feature(
        if_feature,
        title="Instantaneous Frequency",
        cmap="coolwarm"
    )


def plot_modgd(
    modgd
):

    plot_feature(
        modgd,
        title="Modified Group Delay",
        cmap="coolwarm"
    )