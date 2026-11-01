"""Image and video helpers that do not import the GUI or inference runtime."""

import math

import cv2
import numpy as np

from .geometry import scaled_size


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
    file_path: str,
    file_data: np.ndarray,
    file_extension: str = ".jpg",
    jpeg_quality: int = 95,
    png_compression: int = 1,
) -> None:
    """Encode before opening the destination, and reject failed encodes."""
    encode_params = []
    extension = file_extension.lower()
    if extension in (".jpg", ".jpeg"):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    elif extension == ".png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
    try:
        success, encoded = cv2.imencode(file_extension, file_data, encode_params)
    except cv2.error as error:
        raise ValueError(
            f"Unable to encode image as {file_extension}: {file_path}"
        ) from error
    if not success or encoded is None or encoded.size == 0:
        raise ValueError(f"Unable to encode image as {file_extension}: {file_path}")
    with open(file_path, "wb") as file:
        file.write(encoded.tobytes())


def sanitize_fps(frame_rate: float) -> float:
    """Use 30 fps when frame-rate metadata is missing, nonfinite, or nonpositive."""
    try:
        frame_rate = float(frame_rate)
    except (TypeError, ValueError, OverflowError):
        return 30.0
    return frame_rate if math.isfinite(frame_rate) and frame_rate > 0 else 30.0


def get_video_fps(video_path: str) -> float:
    """Reject unreadable videos, sanitize their fps, and always release capture."""
    video_capture = cv2.VideoCapture(video_path)
    try:
        if not video_capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        return sanitize_fps(video_capture.get(cv2.CAP_PROP_FPS))
    finally:
        video_capture.release()


def get_image_resolution(image: np.ndarray) -> tuple[int, int]:
    """Return image dimensions as ``(height, width)``."""
    return image.shape[0], image.shape[1]


def get_image_mode(image: np.ndarray) -> str | None:
    """Describe supported channel layouts without converting the image."""
    shape = image.shape
    if len(shape) == 2:
        return "Grayscale"
    if len(shape) == 3 and shape[2] == 3:
        return "RGB"
    if len(shape) == 3 and shape[2] == 4:
        return "RGBA"
    return None


def scaled_dimensions(image: np.ndarray, factor: float) -> tuple[int, int]:
    """Truncate scaled dimensions and round odd values up, even at unit scale."""
    height, width = get_image_resolution(image)
    return scaled_size(width, height, factor)


def resize_with_factor(
    image: np.ndarray, factor: float, interpolation: int
) -> np.ndarray:
    """Resize to even ``(width, height)`` dimensions with the selected filter."""
    new_width, new_height = scaled_dimensions(image, factor)
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def resize_with_output_factor(
    image: np.ndarray, output_resize_factor: float
) -> np.ndarray:
    """Apply the output scale with linear interpolation and even dimensions."""
    return resize_with_factor(image, output_resize_factor, cv2.INTER_LINEAR)


def sharpen_image(upscaled_image: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp-mask only AI output, falling back unchanged on OpenCV errors."""
    try:
        blurred = cv2.GaussianBlur(upscaled_image, (0, 0), sigmaX=1.2)
        return cv2.addWeighted(
            upscaled_image, 1 + amount, blurred, -amount, 0, dtype=cv2.CV_8U
        )
    except cv2.error:
        return upscaled_image


def sharpen_and_save(
    target_path: str,
    upscaled_image: np.ndarray,
    amount: float,
    file_extension: str = ".jpg",
) -> None:
    """Save sharpened output; encoding and filesystem errors still propagate."""
    image_write(target_path, sharpen_image(upscaled_image, amount), file_extension)


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
