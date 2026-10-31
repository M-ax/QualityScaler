"""Image and video helpers that do not import the GUI or inference runtime."""

import math
import os
import subprocess

import cv2
import numpy as np

from .paths import find_by_relative_path


EXIFTOOL_EXE_PATH = find_by_relative_path(os.path.join("Assets", "exiftool.exe"))


def image_read(file_path: str) -> np.ndarray:
    """Decode an image without OpenCV's platform-dependent filename handling."""
    with open(file_path, "rb") as file:
        encoded = np.frombuffer(file.read(), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"Unable to decode image: {file_path}")
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        raise ValueError(f"Unable to decode image: {file_path}") from error
    if image is None:
        raise ValueError(f"Unable to decode image: {file_path}")
    return image


def image_write(
    file_path: str, file_data: np.ndarray, file_extension: str = ".jpg"
) -> None:
    """Encode before opening the destination, and reject failed encodes."""
    try:
        success, encoded = cv2.imencode(file_extension, file_data)
    except cv2.error as error:
        raise ValueError(
            f"Unable to encode image as {file_extension}: {file_path}"
        ) from error
    if not success or encoded is None or encoded.size == 0:
        raise ValueError(f"Unable to encode image as {file_extension}: {file_path}")
    with open(file_path, "wb") as file:
        file.write(encoded.tobytes())


def copy_file_metadata(original_file_path: str, upscaled_file_path: str) -> None:
    """Best-effort tag copy; an unavailable or failing exiftool is optional."""
    print("[EXIFTOOL] exporting original file tags")
    command = [
        EXIFTOOL_EXE_PATH,
        "-fast",
        "-TagsFromFile",
        original_file_path,
        "-overwrite_original",
        "-all:all",
        "-unsafe",
        "-largetags",
        upscaled_file_path,
    ]
    try:
        subprocess.run(command, check=True, shell=False)
    except (OSError, subprocess.CalledProcessError):
        pass


def get_video_fps(video_path: str) -> float:
    """Read a finite, positive frame rate and always release the capture."""
    video_capture = cv2.VideoCapture(video_path)
    try:
        frame_rate = float(video_capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError(f"Invalid video frame rate for {video_path}: {frame_rate}")
        return frame_rate
    finally:
        video_capture.release()


def get_image_resolution(image: np.ndarray) -> tuple[int, int]:
    """Return image dimensions as ``(height, width)``."""
    return image.shape[0], image.shape[1]


def _add_alpha_channel(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 3:
        alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate((image, alpha), axis=2)
    return image


def blend_images(
    starting_image: np.ndarray,
    upscaled_image: np.ndarray,
    starting_image_importance: float,
) -> np.ndarray:
    """Blend into the upscaled dimensions without mutating either input.

    Keep the original helper's linear resizing, alpha-channel handling, and
    fallback to the upscaled image when the image modes cannot be blended.
    The old downsize call passed INTER_AREA as OpenCV's ``dst`` argument, so
    it also used linear interpolation rather than area interpolation.
    """
    target_height, target_width = get_image_resolution(upscaled_image)
    starting_image = cv2.resize(starting_image, (target_width, target_height))
    try:
        if starting_image.ndim == 3 and starting_image.shape[2] == 4:
            starting_image = _add_alpha_channel(starting_image)
            upscaled_image = _add_alpha_channel(upscaled_image)
        return cv2.addWeighted(
            starting_image,
            starting_image_importance,
            upscaled_image,
            1 - starting_image_importance,
            0,
        )
    except (cv2.error, IndexError):
        return upscaled_image


def blend_images_and_save(
    target_path: str,
    starting_image: np.ndarray,
    upscaled_image: np.ndarray,
    starting_image_importance: float,
    file_extension: str = ".jpg",
) -> None:
    """Save the result of the pure blending helper using the selected codec."""
    image_write(
        target_path,
        blend_images(starting_image, upscaled_image, starting_image_importance),
        file_extension,
    )
