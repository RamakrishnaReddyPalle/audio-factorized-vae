# src/utils/file_utils.py

from pathlib import Path


def replace_root_and_extension(
    source_path,
    source_root,
    target_root,
    new_extension=".wav"
):
    """
    Preserve folder structure while changing root and extension.
    """

    source_path = Path(source_path)

    relative = source_path.relative_to(source_root)

    target_path = Path(target_root) / relative

    target_path = target_path.with_suffix(new_extension)

    return target_path


def ensure_parent(path):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )