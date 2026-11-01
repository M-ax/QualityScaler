"""GUI-free, single-frame previews of processing before output encoding.

No output files or frame cache are created. In particular, previews do not
simulate the video pipeline's intermediate JPEGs, codec artifacts, or YUV
conversion. OpenCV resizing approximates FFmpeg's video scaling filters.
Images retain the backend's channel order (BGR/BGRA or grayscale), not Tk's RGB.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from .ai import AI_upscale, get_model_upscale_factor
from .config import ProcessingConfig
from .formats import VIDEO_EXTENSIONS
from .geometry import VideoGeometry
from .media import image_read, resize_with_output_factor, sanitize_fps, sharpen_image


@dataclass(frozen=True)
class PreviewSource:
    """Readable source metadata; frame positions are zero-based.

    A still image has one frame and the sanitized fallback rate of 30 fps.
    Video frame counts come from the decoder's metadata, never a full scan.
    """

    path: str
    frame_count: int
    fps: float
    is_video: bool


def probe_source(path: str) -> PreviewSource:
    """Validate readability and metadata, releasing any capture before returning.

    Use the same case-insensitive video extensions as batch processing, including
    animated GIFs. Unknown/non-video extensions go through the still-image
    decoder. Videos without a finite positive frame count cannot be seeked
    reliably and are rejected rather than scanned or silently treated as stills.
    """
    path = str(path)
    if Path(path).suffix.lower() not in VIDEO_EXTENSIONS:
        image_read(path)
        return PreviewSource(path, 1, sanitize_fps(0), False)

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {path}")
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(count) or count < 1:
            raise ValueError(f"Unable to determine video frame count: {path}")
        fps = sanitize_fps(capture.get(cv2.CAP_PROP_FPS))
        success, frame = capture.read()
        if not success or frame is None or frame.size == 0:
            raise ValueError(f"Unable to read video: {path}")
        return PreviewSource(path, int(count), fps, True)
    finally:
        capture.release()


def create_preview_engine(config: ProcessingConfig) -> AI_upscale:
    """Create a lazy engine for one worker to reuse, without loading ONNX Runtime.

    Cache by (selected_AI_model, selected_gpu, input_resize_factor,
    tiles_resolution). Output scale and sharpening do not change the engine.
    Like production engines, this instance must not be used concurrently.
    """
    return AI_upscale(
        config.selected_AI_model, config.selected_gpu,
        config.input_resize_factor, config.tiles_resolution,
    )


def _read_video_frame(source: PreviewSource, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(source.path)
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {source.path}")
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise ValueError(f"Unable to seek video frame {frame_index}: {source.path}")
        success, frame = capture.read()
        if not success or frame is None or frame.size == 0:
            raise ValueError(f"Unable to read video frame {frame_index}: {source.path}")
        return frame
    finally:
        capture.release()


def render_preview(
    source: PreviewSource,
    frame_index: int,
    config: ProcessingConfig,
    engine: AI_upscale | None = None,
) -> np.ndarray:
    """Decode and process one clamped frame, returning pre-encode backend pixels.

    Still-image input resizing, inference, and tiling use AI_orchestration.
    Video input uses production's inference geometry (at least two pixels per
    axis), then bypasses the engine's input resize to avoid double scaling.
    The caller supplying an engine is responsible for matching its cache key to
    config. Still images resize before sharpening; videos sharpen native AI
    output before their final, aspect-derived output resize, as production does.
    Each video request opens, seeks, reads once, and releases its capture.
    """
    if source.frame_count < 1:
        raise ValueError("Preview source must have at least one frame")
    frame_index = max(0, min(int(frame_index), source.frame_count - 1))
    original = (
        _read_video_frame(source, frame_index)
        if source.is_video else image_read(source.path)
    )
    if engine is None:
        engine = create_preview_engine(config)
    amount = config.selected_sharpening_amount
    if source.is_video:
        geometry = VideoGeometry.calculate(
            original.shape[1], original.shape[0], config.input_resize_factor,
            get_model_upscale_factor(config.selected_AI_model),
            config.output_resize_factor,
        )
        inference_input = cv2.resize(
            original, geometry.inference_size, interpolation=cv2.INTER_AREA
        )
        upscaled = engine.AI_orchestration(inference_input, input_is_scaled=True)
        if amount:
            upscaled = sharpen_image(upscaled, amount)
        # FFmpeg's default output scale is bicubic; no JPEG/YUV roundtrip here.
        return cv2.resize(upscaled, geometry.output_size, interpolation=cv2.INTER_CUBIC)
    upscaled = engine.AI_orchestration(original)
    upscaled = resize_with_output_factor(upscaled, config.output_resize_factor)
    return sharpen_image(upscaled, amount) if amount else upscaled
