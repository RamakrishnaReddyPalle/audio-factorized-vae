# src/preprocessing/feature_extractors/feature_pipeline.py

from src.preprocessing.feature_extractors.magnitude_phase import (
    MagnitudePhaseExtractor
)

from src.preprocessing.feature_extractors.instantaneous_frequency import (
    InstantaneousFrequencyExtractor
)

from src.preprocessing.feature_extractors.modgd import (
    MODGDExtractor
)

from src.preprocessing.feature_extractors.logmel import (
    LogMelExtractor
)

from src.preprocessing.feature_extractors.multiresolution import (
    MultiResolutionExtractor
)


class FeaturePipeline:

    def __init__(
        self,
        cfg
    ):

        fe_cfg = cfg[
            "feature_extraction"
        ]

        stft_cfg = fe_cfg[
            "stft"
        ]

        self.mag_phase = (
            MagnitudePhaseExtractor(

                n_fft=
                    stft_cfg["n_fft"],

                hop_length=
                    stft_cfg["hop_length"],

                win_length=
                    stft_cfg["win_length"],

                window=
                    stft_cfg["window"]
            )
        )

        self.if_extractor = (
            InstantaneousFrequencyExtractor(

                hop_length=
                    stft_cfg["hop_length"],

                sample_rate=
                    fe_cfg["sample_rate"]
            )
        )

        self.modgd = (
            MODGDExtractor(

                alpha=
                    fe_cfg["modgd"]["alpha"],

                gamma=
                    fe_cfg["modgd"]["gamma"]
            )
        )

        mel_cfg = fe_cfg["mel"]

        self.logmel = (
            LogMelExtractor(

                sample_rate=
                    fe_cfg["sample_rate"],

                n_fft=
                    stft_cfg["n_fft"],

                hop_length=
                    stft_cfg["hop_length"],

                win_length=
                    stft_cfg["win_length"],

                n_mels=
                    mel_cfg["n_mels"],

                fmin=
                    mel_cfg["fmin"],

                fmax=
                    mel_cfg["fmax"]
            )
        )

        mr_cfg = fe_cfg[
            "multiresolution"
        ]

        self.multires = (
            MultiResolutionExtractor(

                sample_rate=
                    fe_cfg["sample_rate"],

                fft_sizes=
                    mr_cfg["fft_sizes"],

                hop_lengths=
                    mr_cfg["hop_lengths"],

                win_lengths=
                    mr_cfg["win_lengths"]
            )
        )

    def extract(
        self,
        waveform
    ):

        mp = self.mag_phase.extract(
            waveform
        )

        features = {

            **mp,

            "if":
                self.if_extractor.extract(
                    mp["phase"]
                ),

            "modgd":
                self.modgd.extract(
                    mp["stft"]
                ),

            "logmel":
                self.logmel.extract(
                    waveform
                )
        }

        features.update(

            self.multires.extract(
                waveform
            )
        )

        return features