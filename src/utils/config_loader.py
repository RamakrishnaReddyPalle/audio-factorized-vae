# src/utils/config_loader.py

from pathlib import Path
import yaml


def load_yaml(config_path):
    """
    Load YAML config file.
    """

    config_path = Path(config_path)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg