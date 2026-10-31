"""Serializable processing choices shared by the desktop and job backend.

This module deliberately has no GUI, GPU, media, or storage imports.
"""

from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    selected_file_list: list[str]
    selected_output_path: str
    selected_AI_model: str
    selected_AI_multithreading: int
    input_resize_factor: float
    output_resize_factor: float
    selected_gpu: str
    tiles_resolution: int
    selected_sharpening_amount: float
    selected_keep_frames: bool
    selected_image_extension: str
    selected_video_extension: str
    selected_video_codec: str
