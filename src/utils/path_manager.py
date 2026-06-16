# src/utils/path_manager.py

from pathlib import Path


class PathManager:

    def __init__(self, project_root, config):

        self.project_root = Path(project_root)

        self.config = config

    def get_path(self, relative_path):

        return self.project_root / relative_path

    def create_required_dirs(self):

        dirs = []

        dirs.append(
            self.get_path(
                self.config["dataset"]["metadata_dir"]
            )
        )

        dirs.append(
            self.get_path(
                self.config["outputs"]["figures_dir"]
            )
        )

        dirs.append(
            self.get_path(
                self.config["outputs"]["checkpoint_dir"]
            )
        )

        dirs.append(
            self.get_path(
                self.config["outputs"]["swap_dir"]
            )
        )

        for d in dirs:

            d.mkdir(
                parents=True,
                exist_ok=True
            )