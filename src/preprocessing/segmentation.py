# src/preprocessing/segmentation.py

import numpy as np

from scipy.signal import find_peaks
from scipy.ndimage import median_filter

import librosa


class AcousticSyllableSegmenter:

    def __init__(
        self,
        sr=16000,
        frame_length=512,
        hop_length=128,
        smoothing_window=9,
        min_fragment_ms=80,
        max_fragment_ms=600,
        min_valley_distance_ms=80
    ):

        self.sr = sr

        self.frame_length = frame_length

        self.hop_length = hop_length

        self.smoothing_window = smoothing_window

        self.min_samples = int(
            sr * min_fragment_ms / 1000
        )

        self.max_samples = int(
            sr * max_fragment_ms / 1000
        )

        self.min_valley_distance_frames = max(
            1,
            int(
                (
                    min_valley_distance_ms / 1000
                )
                * sr
                / hop_length
            )
        )

    def compute_envelope(
        self,
        waveform
    ):

        rms = librosa.feature.rms(
            y=waveform,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        rms = median_filter(
            rms,
            size=self.smoothing_window
        )

        return rms

    def find_boundaries(
        self,
        waveform
    ):

        envelope = self.compute_envelope(
            waveform
        )

        valleys, _ = find_peaks(
            -envelope,
            distance=self.min_valley_distance_frames
        )

        boundaries = (
            valleys
            * self.hop_length
        )

        return boundaries, envelope

    def segment_interval(
        self,
        waveform,
        start,
        end
    ):

        speech = waveform[start:end]

        boundaries, envelope = (
            self.find_boundaries(
                speech
            )
        )

        boundaries = (
            boundaries + start
        )

        cuts = [start]

        for b in boundaries:

            if (
                start < b < end
            ):
                cuts.append(
                    int(b)
                )

        cuts.append(end)

        cuts = sorted(
            list(set(cuts))
        )

        fragments = []

        for i in range(
            len(cuts) - 1
        ):

            s = cuts[i]
            e = cuts[i + 1]

            length = e - s

            if (
                length
                < self.min_samples
            ):
                continue

            if (
                length
                > self.max_samples
            ):

                current = s

                while current < e:

                    nxt = min(
                        current
                        + self.max_samples,
                        e
                    )

                    if (
                        nxt - current
                        >= self.min_samples
                    ):
                        fragments.append(
                            (
                                current,
                                nxt
                            )
                        )

                    current = nxt

            else:

                fragments.append(
                    (
                        s,
                        e
                    )
                )

        return fragments, envelope