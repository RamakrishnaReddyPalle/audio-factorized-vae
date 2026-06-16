# src/models/factorized/feature_group_manager.py


class FeatureGroupManager:

    def __init__(
        self
    ):
        pass

    def _ensure_channel_dim(
        self,
        x
    ):

        # Dataset batch
        #
        # [B,F,T]
        #
        # Encoder expects
        #
        # [B,1,F,T]

        if x.dim() == 3:

            x = x.unsqueeze(
                1
            )

        return x

    def build_groups(
        self,
        batch
    ):

        groups = {}

        # --------------------------------------------------
        # Content
        # --------------------------------------------------

        groups["content"] = {

            "logmel":
                self._ensure_channel_dim(
                    batch["logmel"]
                ),

            "mr_mag_256":
                self._ensure_channel_dim(
                    batch["mr_mag_256"]
                ),

            "if":
                self._ensure_channel_dim(
                    batch["if"]
                )
        }

        # --------------------------------------------------
        # Speaker
        # --------------------------------------------------

        groups["speaker"] = {

            "mr_mag_512":
                self._ensure_channel_dim(
                    batch["mr_mag_512"]
                ),

            "mr_mag_256":
                self._ensure_channel_dim(
                    batch["mr_mag_256"]
                ),

            "logmel":
                self._ensure_channel_dim(
                    batch["logmel"]
                )
        }

        # --------------------------------------------------
        # Environment
        # --------------------------------------------------

        groups["environment"] = {

            "magnitude":
                self._ensure_channel_dim(
                    batch["magnitude"]
                ),

            "mr_mag_1024":
                self._ensure_channel_dim(
                    batch["mr_mag_1024"]
                ),

            "if":
                self._ensure_channel_dim(
                    batch["if"]
                )
        }

        # --------------------------------------------------
        # Excitation
        # --------------------------------------------------

        groups["excitation"] = {

            "modgd":
                self._ensure_channel_dim(
                    batch["modgd"]
                )
        }

        # --------------------------------------------------
        # Fidelity
        # --------------------------------------------------

        groups["fidelity"] = {

            "phase_sin":
                self._ensure_channel_dim(
                    batch["phase_sin"]
                ),

            "phase_cos":
                self._ensure_channel_dim(
                    batch["phase_cos"]
                ),

            "mr_mag_512":
                self._ensure_channel_dim(
                    batch["mr_mag_512"]
                ),

            "mr_mag_1024":
                self._ensure_channel_dim(
                    batch["mr_mag_1024"]
                ),

            "magnitude":
                self._ensure_channel_dim(
                    batch["magnitude"]
                ),

            "modgd":
                self._ensure_channel_dim(
                    batch["modgd"]
                )
        }

        return groups