# src/models/factorized/latent_swapper.py

import copy


class LatentSwapper:

    @staticmethod
    def swap_environment(
        source_latents,
        target_latents
    ):
        """
        Environment transfer.

        source = clean S1

        target = noisy S2
        """

        out = copy.deepcopy(
            target_latents
        )

        out["environment"] = copy.deepcopy(
            source_latents["environment"]
        )

        return out

    @staticmethod
    def swap_fidelity(
        source_latents,
        target_latents
    ):
        """
        Optional fidelity transfer.
        """

        out = copy.deepcopy(
            target_latents
        )

        out["fidelity"] = copy.deepcopy(
            source_latents["fidelity"]
        )

        return out

    @staticmethod
    def custom_swap(
        source_latents,
        target_latents,
        keys
    ):
        """
        Arbitrary latent swapping.
        """

        out = copy.deepcopy(
            target_latents
        )

        for key in keys:

            out[key] = copy.deepcopy(
                source_latents[key]
            )

        return out