"""AI image upscaling without GUI imports or eager ONNX Runtime loading."""

import os

import cv2
import numpy as np

from .gpu import get_AI_providers
from .media import get_image_mode, get_image_resolution, resize_with_factor, scaled_dimensions
from .paths import find_by_relative_path


def get_model_upscale_factor(selected_AI_model: str) -> int:
    """Read the native upscale factor from a model name, defaulting to 1x."""
    for factor in (1, 2, 3, 4):
        if f"x{factor}" in selected_AI_model:
            return factor
    return 1


def calculate_optimal_workers(image, model, input_resize_factor, max_resolution, selected):
    """Size video concurrency without allocating a GPU session or resized image."""
    width, height = scaled_dimensions(image, input_resize_factor)
    if width <= 0 or height <= 0:
        raise ValueError("Input resize factor produces an empty frame")
    frames_simultaneously = max_resolution * max_resolution // (height * width)
    print(f"Frames supported simultaneously by GPU with {model}: {frames_simultaneously}")
    workers = max(1, min(frames_simultaneously, selected))
    print(f"Selected threads number: {workers}")
    return workers


class AI_upscale:
    """Resize input and upscale at the model's native scale, loading on demand.

    Each engine owns its session and mutable inference input buffer. Output
    resizing belongs to the media pipeline, not to this engine.
    """

    def __init__(
        self,
        selected_AI_model: str,
        selected_gpu: str,
        input_resize_factor: float,
        tiles_resolution: int,
    ):
        self.selected_AI_model = selected_AI_model
        self.selected_gpu = selected_gpu
        self.input_resize_factor = input_resize_factor
        self.tiles_resolution = tiles_resolution
        self.selected_AI_model_path = find_by_relative_path(
            os.path.join("AI-onnx", f"{selected_AI_model}_fp16.onnx")
        )
        self.upscale_factor = get_model_upscale_factor(selected_AI_model)
        self.inferenceSession = None
        self.input_name = None
        self.onnx_input = None

    def _load_inferenceSession(self):
        # Keep ORT out of backend imports, worker sizing, and engine construction.
        from onnxruntime import (
            ExecutionMode, GraphOptimizationLevel, InferenceSession, SessionOptions,
        )

        providers = get_AI_providers()
        provider_options = []
        for provider in providers:
            if provider == "DmlExecutionProvider":
                if self.selected_gpu == "Auto":
                    options = {"performance_preference": "high_performance"}
                else:
                    device_id = str(self.selected_gpu)
                    # Saved legacy selections were one-based ("GPU 1"/"GPU1").
                    if device_id.startswith("GPU"):
                        device_id = str(int(device_id[3:].strip()) - 1)
                    options = {"device_id": device_id}
                provider_options.append(options)
            else:
                provider_options.append({})

        session_options = SessionOptions()
        session_options.enable_profiling = False
        session_options.intra_op_num_threads = 1
        session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        # DirectML does not support memory patterns or parallel graph execution.
        # These options govern one session, not concurrency between sessions.
        if "DmlExecutionProvider" in providers:
            session_options.enable_mem_pattern = False
            session_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        return InferenceSession(
            path_or_bytes=self.selected_AI_model_path,
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )

    def ensure_session(self):
        """Return the session, initializing its cached input metadata once."""
        if self.inferenceSession is None:
            session = self._load_inferenceSession()
            input_name = session.get_inputs()[0].name
            self.inferenceSession = session
            self.input_name = input_name
            self.onnx_input = {input_name: None}
        elif self.input_name is None or self.onnx_input is None:
            # Also support caller-supplied sessions (e.g. CPU probes and fakes).
            self.input_name = self.inferenceSession.get_inputs()[0].name
            self.onnx_input = {self.input_name: None}
        return self.inferenceSession

    def calculate_target_resolution(self, image: np.ndarray) -> tuple[int, int]:
        height, width = get_image_resolution(image)
        return height * self.upscale_factor, width * self.upscale_factor

    def resize_with_input_factor(self, image: np.ndarray) -> np.ndarray:
        return resize_with_factor(image, self.input_resize_factor, cv2.INTER_AREA)

    def calculate_optimal_multithreads_number(
        self, video_frame: np.ndarray, selected_AI_multithreading: int
    ) -> int:
        """Size concurrency from an already decoded frame, not a frame file."""
        return calculate_optimal_workers(
            video_frame, self.selected_AI_model, self.input_resize_factor,
            self.tiles_resolution, selected_AI_multithreading,
        )

    def image_need_tilling(self, image: np.ndarray) -> bool:
        height, width = get_image_resolution(image)
        return height * width > self.tiles_resolution * self.tiles_resolution

    def calculate_tiles_number(self, image: np.ndarray) -> tuple[int, int]:
        height, width = get_image_resolution(image)
        tiles_x = (width + self.tiles_resolution - 1) // self.tiles_resolution
        tiles_y = (height + self.tiles_resolution - 1) // self.tiles_resolution
        return tiles_x, tiles_y

    def normalize_image(self, image: np.ndarray, original_dtype) -> tuple[np.ndarray, float]:
        # Most sources are uint8; avoid a full max() scan on every tile.
        if original_dtype == np.uint8:
            max_range = 255.0
        else:
            max_range = 65535.0 if np.max(image) > 256 else 255.0
        image /= max_range
        return image, max_range

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
        return np.expand_dims(image, axis=0)

    def onnxruntime_inference(self, image: np.ndarray) -> np.ndarray:
        self.ensure_session()
        self.onnx_input[self.input_name] = image
        return self.inferenceSession.run(None, self.onnx_input)[0]

    def postprocess_output(self, onnx_output: np.ndarray) -> np.ndarray:
        onnx_output = np.squeeze(onnx_output, axis=0)
        onnx_output = np.clip(onnx_output, 0, 1)
        return np.transpose(onnx_output, (1, 2, 0))

    def de_normalize_image(
        self, onnx_output: np.ndarray, max_range: float
    ) -> np.ndarray:
        onnx_output *= max_range
        return onnx_output.astype(np.uint8) if max_range == 255 else onnx_output.astype(np.float32)

    def AI_upscale(self, image: np.ndarray) -> np.ndarray:
        original_dtype = image.dtype
        # Normalization is in-place. Own the float buffer so overlapping tiles
        # cannot normalize a float32 source (or their shared context) repeatedly.
        image = image.astype(np.float32)
        image_mode = get_image_mode(image)
        image, max_range = self.normalize_image(image, original_dtype)

        match image_mode:
            case "RGB":
                image = self.preprocess_image(image)
                onnx_output = self.onnxruntime_inference(image)
                onnx_output = self.postprocess_output(onnx_output)
                return self.de_normalize_image(onnx_output, max_range)

            case "RGBA":
                alpha = image[:, :, 3]
                image = image[:, :, :3]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = self.preprocess_image(image)
                onnx_output_image = self.onnxruntime_inference(image)
                onnx_output_image = self.postprocess_output(onnx_output_image)
                onnx_output_image = cv2.cvtColor(
                    onnx_output_image, cv2.COLOR_BGR2RGBA
                )

                alpha = np.expand_dims(alpha, axis=-1)
                alpha = np.repeat(alpha, 3, axis=-1)
                alpha = self.preprocess_image(alpha)
                onnx_output_alpha = self.onnxruntime_inference(alpha)
                onnx_output_alpha = self.postprocess_output(onnx_output_alpha)
                onnx_output_alpha = cv2.cvtColor(
                    onnx_output_alpha, cv2.COLOR_RGB2GRAY
                )

                onnx_output_image[:, :, 3] = onnx_output_alpha
                return self.de_normalize_image(onnx_output_image, max_range)

            case "Grayscale":
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                image = self.preprocess_image(image)
                onnx_output = self.onnxruntime_inference(image)
                onnx_output = self.postprocess_output(onnx_output)
                output_image = cv2.cvtColor(onnx_output, cv2.COLOR_RGB2GRAY)
                return self.de_normalize_image(output_image, max_range)

            case _:
                return image

    def _tile_feather_weights(self, length: int, ramp: int) -> np.ndarray:
        """Positive linear edge ramps, capped so the ramps cannot overlap."""
        weights = np.ones(length, dtype=np.float32)
        r = min(ramp, length // 2)
        if r > 0:
            edge = np.arange(1, r + 1, dtype=np.float32) / (r + 1)
            weights[:r] = edge
            weights[length - r:] = edge[::-1]
        return weights

    def AI_upscale_with_tilling(self, image: np.ndarray) -> np.ndarray:
        overlap = 16
        tiles_x, tiles_y = self.calculate_tiles_number(image)
        img_height, img_width = get_image_resolution(image)
        t_height, t_width = self.calculate_target_resolution(image)
        uf = self.upscale_factor
        ramp = overlap * uf
        tile_w = img_width // tiles_x
        tile_h = img_height // tiles_y

        pad_cfg = ((overlap, overlap), (overlap, overlap))
        if image.ndim == 3:
            pad_cfg += ((0, 0),)
        padded = np.pad(image, pad_cfg, mode="reflect")
        image_mode = get_image_mode(image)
        channels = 1 if image_mode == "Grayscale" else (4 if image_mode == "RGBA" else 3)
        accumulator = np.zeros(
            (t_height + 2 * ramp, t_width + 2 * ramp, channels), dtype=np.float32,
        )
        weights = np.zeros(
            (t_height + 2 * ramp, t_width + 2 * ramp, 1), dtype=np.float32,
        )

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                cx0, cy0 = tx * tile_w, ty * tile_h
                # Give the final row/column all remaining pixels.
                cx1 = img_width if tx == tiles_x - 1 else (tx + 1) * tile_w
                cy1 = img_height if ty == tiles_y - 1 else (ty + 1) * tile_h
                tile = padded[cy0:cy1 + 2 * overlap, cx0:cx1 + 2 * overlap]
                upscaled_tile = self.AI_upscale(tile)
                if upscaled_tile.ndim == 2:
                    upscaled_tile = upscaled_tile[:, :, None]
                th, tw = upscaled_tile.shape[:2]
                wy = self._tile_feather_weights(th, ramp)
                wx = self._tile_feather_weights(tw, ramp)
                mask = (wy[:, None] * wx[None, :])[:, :, None]

                # The accumulator includes the context border; writes never
                # need clipping, including the uneven final row and column.
                y0, x0 = cy0 * uf, cx0 * uf
                accumulator[y0:y0 + th, x0:x0 + tw] += upscaled_tile.astype(np.float32) * mask
                weights[y0:y0 + th, x0:x0 + tw] += mask

        accumulator = accumulator[ramp:ramp + t_height, ramp:ramp + t_width]
        weights = weights[ramp:ramp + t_height, ramp:ramp + t_width]
        np.maximum(weights, 1e-6, out=weights)
        blended = accumulator / weights + 0.5
        if image_mode == "Grayscale":
            return blended[:, :, 0].astype(np.uint8)
        return blended.astype(np.uint8)

    def AI_orchestration(self, image: np.ndarray) -> np.ndarray:
        self.ensure_session()
        resized_image = self.resize_with_input_factor(image)
        if self.image_need_tilling(resized_image):
            return self.AI_upscale_with_tilling(resized_image)
        return self.AI_upscale(resized_image)
