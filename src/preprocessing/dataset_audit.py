# src/preprocessing/dataset_audit.py

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.utils.audio_loader import AudioLoader


class DatasetAuditor:

    def __init__(self, raw_dir, extensions):

        self.raw_dir = Path(raw_dir)

        self.extensions = extensions

    def get_audio_files(self):

        files = []

        for ext in self.extensions:

            files.extend(
                self.raw_dir.rglob(f"*{ext}")
            )

        return sorted(files)

    @staticmethod
    def parse_labels(filepath):

        filename = filepath.name.lower()

        parent = filepath.parent.name.lower()

        split = "train"

        if parent == "test_samples":
            split = "test"

        speaker = "unknown"

        if "s1_" in filename:
            speaker = "s1"

        elif "s2_" in filename:
            speaker = "s2"

        condition = "unknown"

        if "clean" in filename:
            condition = "clean"

        elif "noisy" in filename:
            condition = "noisy"

        return split, speaker, condition

    def build_inventory(self):

        files = self.get_audio_files()

        rows = []

        print(f"\nFound {len(files)} audio files\n")

        for filepath in tqdm(files):

            try:

                y, sr = AudioLoader.load(filepath)

                mono = AudioLoader.to_mono(y)

                split, speaker, condition = (
                    self.parse_labels(filepath)
                )

                channels = 1

                if y.ndim > 1:
                    channels = y.shape[0]

                rows.append(
                    {
                        "filepath": str(filepath),
                        "filename": filepath.name,
                        "split": split,
                        "speaker": speaker,
                        "condition": condition,
                        "extension": filepath.suffix,
                        "sample_rate": sr,
                        "channels": channels,
                        "duration_sec": len(mono) / sr,
                        "num_samples": len(mono),
                        "rms": AudioLoader.rms(mono),
                        "peak": AudioLoader.peak(mono),
                    }
                )

            except Exception as e:

                print(f"\nERROR : {filepath}")
                print(e)

        return pd.DataFrame(rows)