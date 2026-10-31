"""Input selection rules, independent of Tk and the inference runtime."""

from itertools import chain
import os
from pathlib import Path

from .formats import INPUT_EXTENSIONS


def supported_input_files(paths):
    """Return existing media files in order, with canonical paths and no duplicates.

    Directories are not expanded. Match the final extension case-insensitively,
    not a substring in the path (``movie.mp4.txt`` is not a video).
    """
    result = []
    seen = set()
    for value in paths:
        try:
            path = Path(value)
            if path.suffix.lower() not in INPUT_EXTENSIONS or not path.is_file():
                continue
            path = path.resolve()
        except (OSError, ValueError, TypeError):
            continue
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            result.append(str(path))
    return result


def merge_input_files(existing, incoming):
    """Append new valid inputs without duplicating existing selections."""
    return supported_input_files(chain(existing, incoming))
