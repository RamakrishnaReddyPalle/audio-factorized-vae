# src/preprocessing/vad.py

import librosa


class VoiceActivityDetector:

    def __init__(
        self,
        top_db=25
    ):
        self.top_db = top_db

    def detect_regions(
        self,
        waveform
    ):

        intervals = librosa.effects.split(
            waveform,
            top_db=self.top_db
        )

        return intervals