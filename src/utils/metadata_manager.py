# src/utils/metadata_manager.py

from pathlib import Path
import pandas as pd
import json


class MetadataManager:

    @staticmethod
    def save_csv(df, filepath):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            filepath,
            index=False
        )

    @staticmethod
    def load_csv(filepath):

        return pd.read_csv(filepath)

    @staticmethod
    def save_json(data, filepath):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(filepath, "w") as f:
            json.dump(
                data,
                f,
                indent=4
            )

    @staticmethod
    def load_json(filepath):

        with open(filepath, "r") as f:
            return json.load(f)