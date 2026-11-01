"""Pixel geometry shared by image resizing and video extraction."""

from dataclasses import dataclass


def scaled_size(width: int, height: int, factor: float) -> tuple[int, int]:
    """Truncate each scaled axis, then round odd dimensions up to even."""
    width, height = int(width * factor), int(height * factor)
    return width + width % 2, height + height % 2


@dataclass(frozen=True)
class VideoGeometry:
    source_size: tuple[int, int]
    inference_size: tuple[int, int]
    output_size: tuple[int, int]

    @classmethod
    def calculate(cls, width, height, input_factor, model_factor, output_factor):
        if width <= 0 or height <= 0:
            raise ValueError("Video dimensions must be positive")
        inference_size = tuple(max(2, size) for size in scaled_size(width, height, input_factor))
        # Preserve 2026's final video geometry: nearest-even width, followed by
        # an aspect-derived height. This intentionally differs from the AI's
        # per-axis truncate/round-up input rule above.
        input_width = max(2, round(width * input_factor / 2) * 2)
        input_height = max(2, round(input_width / (width / height) / 2) * 2)
        scale = model_factor * output_factor
        output_width = max(2, round(input_width * scale / 2) * 2)
        output_height = max(2, round(output_width / (input_width / input_height) / 2) * 2)
        return cls((width, height), inference_size, (output_width, output_height))
