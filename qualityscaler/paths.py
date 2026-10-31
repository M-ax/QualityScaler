"""Resource lookup and stable output names, independent of GUI settings.

Processing settings belong in cache metadata, not in filenames. Legacy positional
arguments are retained; the final setting is now named selected_sharpening_amount.
"""

import os
import sys


OUTPUT_PATH_CODED = "Same path as input files"


def find_by_relative_path(relative_path: str) -> str:
    """Resolve bundled resources against the repository root or PyInstaller."""
    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = getattr(sys, "_MEIPASS", repository_root)
    return os.path.join(base_path, relative_path)


def _output_directory(source_path: str, selected_output_path: str) -> str:
    if selected_output_path == OUTPUT_PATH_CODED:
        return os.path.dirname(source_path)
    return selected_output_path


def _output_filename(
    source_path: str, selected_output_path: str, selected_extension: str
) -> str:
    stem, _ = os.path.splitext(os.path.basename(source_path))
    return os.path.join(
        _output_directory(source_path, selected_output_path),
        f"{stem}_upscaled{selected_extension}",
    )


def prepare_output_image_filename(
    image_path: str,
    selected_output_path: str,
    selected_AI_model: str,
    input_resize_factor: float,
    output_resize_factor: float,
    selected_image_extension: str,
    selected_sharpening_amount: float,
) -> str:
    """Return ``<source stem>_upscaled<extension>``; settings do not affect names."""
    return _output_filename(
        image_path, selected_output_path, selected_image_extension
    )


def prepare_output_video_filename(
    video_path: str,
    selected_output_path: str,
    selected_AI_model: str,
    input_resize_factor: float,
    output_resize_factor: float,
    selected_video_extension: str,
    selected_sharpening_amount: float,
) -> str:
    """Return ``<source stem>_upscaled<extension>``; settings live in the cache."""
    return _output_filename(
        video_path, selected_output_path, selected_video_extension
    )


def prepare_output_video_directory_name(
    video_path: str,
    selected_output_path: str,
    selected_AI_model: str,
    input_resize_factor: float,
    output_resize_factor: float,
    selected_sharpening_amount: float,
) -> str:
    """Return a stable ``<source filename>.qualityscaler`` cache directory.

    Including the source extension keeps ``clip.mp4`` and ``clip.mkv`` separate.
    The pipeline records processing settings in SQLite and validates them there.
    """
    return os.path.join(
        _output_directory(video_path, selected_output_path),
        f"{os.path.basename(video_path)}.qualityscaler",
    )
