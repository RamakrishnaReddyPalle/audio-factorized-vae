# src/preprocessing/fragment_dataset_builder.py

from pathlib import Path
import json

import librosa
import numpy as np
import soundfile as sf

from src.preprocessing.vad import (
    VoiceActivityDetector
)

from src.preprocessing.segmentation import (
    AcousticSyllableSegmenter
)


class FragmentDatasetBuilder:

    def __init__(
        self,
        sr,
        vad_top_db,
        min_ms,
        max_ms,
        padded_length=None
    ):

        self.sr = sr

        self.vad = VoiceActivityDetector(
            top_db=vad_top_db
        )

        self.segmenter = (
            AcousticSyllableSegmenter(
                sr=sr,
                min_fragment_ms=min_ms,
                max_fragment_ms=max_ms
            )
        )

    def pad_fragment(
        self,
        fragment,
        padded_length
    ):

        out = np.zeros(
            padded_length,
            dtype=np.float32
        )

        n = min(
            len(fragment),
            padded_length
        )

        out[:n] = fragment[:n]

        return out

    def process_file(
        self,
        audio_path,
        output_dir,
        speaker,
        condition,
        split,
        padded_length
    ):

        waveform, sr = librosa.load(
            audio_path,
            sr=None,
            mono=True
        )

        intervals = self.vad.detect_regions(
            waveform
        )

        metadata = {

            "speaker": speaker,

            "condition": condition,

            "split": split,

            "source_file": str(audio_path),

            "sample_rate": sr,

            "padded_length": padded_length,

            "num_vad_regions": len(intervals),

            "fragments": []
        }

        raw_dir = (
            output_dir
            / "raw"
        )

        padded_dir = (
            output_dir
            / "padded"
        )

        raw_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        padded_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        fragment_id = 0

        for region_idx, (start, end) in enumerate(intervals):

            fragments, envelope = (
                self.segmenter.segment_interval(
                    waveform,
                    start,
                    end
                )
            )

            for s, e in fragments:

                fragment = waveform[s:e]

                raw_path = (
                    raw_dir
                    / f"fragment_{fragment_id:03d}.wav"
                )

                sf.write(
                    raw_path,
                    fragment,
                    sr
                )

                padded_fragment = (
                    self.pad_fragment(
                        fragment,
                        padded_length
                    )
                )

                padded_path = (
                    padded_dir
                    / f"fragment_{fragment_id:03d}.npy"
                )

                np.save(
                    padded_path,
                    padded_fragment
                )

                metadata["fragments"].append({

                    # ---------------------------------
                    # Fragment Identity
                    # ---------------------------------

                    "fragment_id":
                        fragment_id,

                    "position_index":
                        fragment_id,

                    "vad_region_index":
                        region_idx,

                    # ---------------------------------
                    # Temporal Information
                    # ---------------------------------

                    "start_sample":
                        int(s),

                    "end_sample":
                        int(e),

                    "start_time":
                        float(s / sr),

                    "end_time":
                        float(e / sr),

                    "duration":
                        float(
                            (e - s) / sr
                        ),

                    # ---------------------------------
                    # Length Information
                    # ---------------------------------

                    "true_length":
                        int(
                            len(fragment)
                        ),

                    "padded_length":
                        int(
                            padded_length
                        ),

                    "padding_ratio":
                        float(
                            1.0
                            -
                            (
                                len(fragment)
                                /
                                padded_length
                            )
                        ),

                    # ---------------------------------
                    # Storage
                    # ---------------------------------

                    "raw_file":
                        str(raw_path),

                    "tensor_file":
                        str(padded_path)
                })

                fragment_id += 1

        # -----------------------------------------
        # Global Positional Metadata
        # -----------------------------------------

        total_fragments = len(
            metadata["fragments"]
        )

        for fragment in metadata["fragments"]:

            fragment[
                "total_fragments"
            ] = total_fragments

            fragment[
                "relative_position"
            ] = (

                fragment[
                    "position_index"
                ]

                /

                max(
                    total_fragments - 1,
                    1
                )
            )

        metadata[
            "total_fragments"
        ] = total_fragments

        # -----------------------------------------
        # Save Metadata
        # -----------------------------------------

        metadata_path = (
            output_dir
            / "metadata.json"
        )

        with open(
            metadata_path,
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

        return metadata