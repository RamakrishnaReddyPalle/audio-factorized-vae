# src/preprocessing/feature_extractors/feature_fusion.py


class FeatureFusion:

    @staticmethod
    def summarize_shapes(
        features
    ):

        summary = {}

        for k, v in features.items():

            try:

                summary[k] = list(
                    v.shape
                )

            except Exception:

                pass

        return summary