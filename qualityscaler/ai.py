"""AI image upscaling without GUI imports or eager ONNX Runtime loading."""

import os

import cv2
import numpy as np

from .paths import find_by_relative_path


def scaled_dimensions(image, factor):
    """Return even (width, height) dimensions used for non-unit resizing."""
    height, width = image.shape[:2]
    width, height = int(width * factor), int(height * factor)
    return width + width % 2, height + height % 2


def calculate_optimal_workers(image, model, input_resize_factor, max_resolution, selected):
    """Size video concurrency without allocating a GPU session or resized image."""
    if input_resize_factor == 1:
        height, width = image.shape[:2]
    else:
        width, height = scaled_dimensions(image, input_resize_factor)
    if width <= 0 or height <= 0:
        raise ValueError("Input resize factor produces an empty frame")
    frames_simultaneously = max_resolution * max_resolution // (height * width)
    print(f"Frames supported simultaneously by GPU with {model}: {frames_simultaneously}")
    workers = max(1, min(frames_simultaneously, selected))
    if "IRCNN" in model:
        workers = 1
    print(f"Selected threads number: {workers}")
    return workers


class AI_upscale:
    """Run the existing DirectML upscaling, resizing, and tiling workflow."""

    def __init__(
        self,
        AI_model_name: str,
        directml_gpu: str,
        input_resize_factor: float,
        output_resize_factor: float,
        max_resolution: int,
    ):
        self.AI_model_name = AI_model_name
        self.directml_gpu = directml_gpu
        self.input_resize_factor = input_resize_factor
        self.output_resize_factor = output_resize_factor
        self.max_resolution = max_resolution

        self.AI_model_path = find_by_relative_path(
            os.path.join("AI-onnx", f"{self.AI_model_name}_fp16.onnx")
        )
        self.upscale_factor = self._get_upscale_factor()
        self.inferenceSession = self._load_inferenceSession()

    def _get_upscale_factor(self) -> int:
        if "x1" in self.AI_model_name:
            return 1
        elif "x2" in self.AI_model_name:
            return 2
        elif "x4" in self.AI_model_name:
            return 4

    def _load_inferenceSession(self):
        # Import only when constructing a real session, not for module imports
        # or tests that supply a fake inference session.
        from onnxruntime import ExecutionMode, InferenceSession, SessionOptions

        providers = ["DmlExecutionProvider"]
        match self.directml_gpu:
            case "Auto":
                provider_options = [{"performance_preference": "high_performance"}]
            case "GPU 1":
                provider_options = [{"device_id": "0"}]
            case "GPU 2":
                provider_options = [{"device_id": "1"}]
            case "GPU 3":
                provider_options = [{"device_id": "2"}]
            case "GPU 4":
                provider_options = [{"device_id": "3"}]

        # DirectML does not support memory patterns or parallel graph execution.
        # These options govern one session, not concurrency between sessions.
        session_options = SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        return InferenceSession(
            path_or_bytes=self.AI_model_path,
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )

    def get_image_mode(self, image: np.ndarray) -> str:
        shape = image.shape
        if len(shape) == 2:
            return "Grayscale"
        elif len(shape) == 3 and shape[2] == 3:
            return "RGB"
        elif len(shape) == 3 and shape[2] == 4:
            return "RGBA"

    def get_image_resolution(self, image: np.ndarray) -> tuple[int, int]:
        return image.shape[0], image.shape[1]

    def calculate_target_resolution(self, image: np.ndarray) -> tuple[int, int]:
        height, width = self.get_image_resolution(image)
        return height * self.upscale_factor, width * self.upscale_factor

    def resize_with_input_factor(self, image: np.ndarray) -> np.ndarray:
        new_width, new_height = scaled_dimensions(image, self.input_resize_factor)

        if self.input_resize_factor > 1:
            return cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )
        elif self.input_resize_factor < 1:
            return cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
        else:
            return image

    def resize_with_output_factor(self, image: np.ndarray) -> np.ndarray:
        new_width, new_height = scaled_dimensions(image, self.output_resize_factor)

        if self.output_resize_factor > 1:
            return cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )
        elif self.output_resize_factor < 1:
            return cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
        else:
            return image

    def calculate_optimal_multithreads_number(
        self, video_frame: np.ndarray, selected_AI_multithreading: int
    ) -> int:
        """Size concurrency from an already decoded frame, not a frame file."""
        return calculate_optimal_workers(
            video_frame, self.AI_model_name, self.input_resize_factor,
            self.max_resolution, selected_AI_multithreading,
        )

    def image_need_tilling(self, image: np.ndarray) -> bool:
        height, width = self.get_image_resolution(image)
        return height * width > self.max_resolution * self.max_resolution

    def add_alpha_channel(self, image: np.ndarray) -> np.ndarray:
        if image.shape[2] == 3:
            alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
            image = np.concatenate((image, alpha), axis=2)
        return image

    def calculate_tiles_number(self, image: np.ndarray) -> tuple[int, int]:
        height, width = self.get_image_resolution(image)
        tiles_x = (width + self.max_resolution - 1) // self.max_resolution
        tiles_y = (height + self.max_resolution - 1) // self.max_resolution
        return tiles_x, tiles_y

    def split_image_into_tiles(
        self, image: np.ndarray, tiles_x: int, tiles_y: int
    ) -> list[np.ndarray]:
        img_height, img_width = self.get_image_resolution(image)
        tile_width = img_width // tiles_x
        tile_height = img_height // tiles_y
        tiles = []
        for y in range(tiles_y):
            y_start = y * tile_height
            y_end = (y + 1) * tile_height
            for x in range(tiles_x):
                x_start = x * tile_width
                x_end = (x + 1) * tile_width
                tiles.append(image[y_start:y_end, x_start:x_end])
        return tiles

    def combine_tiles_into_image(
        self,
        image: np.ndarray,
        tiles: list[np.ndarray],
        t_height: int,
        t_width: int,
        num_tiles_x: int,
    ) -> np.ndarray:
        match self.get_image_mode(image):
            case "Grayscale":
                tiled_image = np.zeros((t_height, t_width), dtype=np.uint8)
            case "RGB":
                tiled_image = np.zeros((t_height, t_width, 3), dtype=np.uint8)
            case "RGBA":
                tiled_image = np.zeros((t_height, t_width, 4), dtype=np.uint8)

        for tile_index in range(len(tiles)):
            actual_tile = tiles[tile_index]
            tile_height, tile_width = self.get_image_resolution(actual_tile)
            row = tile_index // num_tiles_x
            col = tile_index % num_tiles_x
            y_start = row * tile_height
            y_end = y_start + tile_height
            x_start = col * tile_width
            x_end = x_start + tile_width

            match self.get_image_mode(image):
                case "Grayscale" | "RGB":
                    tiled_image[y_start:y_end, x_start:x_end] = actual_tile
                case "RGBA":
                    tiled_image[y_start:y_end, x_start:x_end] = self.add_alpha_channel(
                        actual_tile
                    )
        return tiled_image

    def normalize_image(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        max_range = 255
        if np.max(image) > 256:
            max_range = 65535
        return image / max_range, max_range

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0)

    def onnxruntime_inference(self, image: np.ndarray) -> np.ndarray:
        onnx_input = {self.inferenceSession.get_inputs()[0].name: image}
        return self.inferenceSession.run(None, onnx_input)[0]

    def postprocess_output(self, onnx_output: np.ndarray) -> np.ndarray:
        onnx_output = np.squeeze(onnx_output, axis=0)
        onnx_output = np.clip(onnx_output, 0, 1)
        return np.transpose(onnx_output, (1, 2, 0))

    def de_normalize_image(
        self, onnx_output: np.ndarray, max_range: int
    ) -> np.ndarray:
        match max_range:
            case 255:
                return (onnx_output * max_range).astype(np.uint8)
            case 65535:
                return (onnx_output * max_range).round().astype(np.float32)

    def AI_upscale(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image_mode = self.get_image_mode(image)
        image, max_range = self.normalize_image(image)

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
                image = image.astype(np.float32)
                alpha = alpha.astype(np.float32)

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

    def AI_upscale_with_tilling(self, image: np.ndarray) -> np.ndarray:
        t_height, t_width = self.calculate_target_resolution(image)
        tiles_x, tiles_y = self.calculate_tiles_number(image)
        tiles_list = self.split_image_into_tiles(image, tiles_x, tiles_y)
        tiles_list = [self.AI_upscale(tile) for tile in tiles_list]
        return self.combine_tiles_into_image(
            image, tiles_list, t_height, t_width, tiles_x
        )

    def AI_orchestration(self, image: np.ndarray) -> np.ndarray:
        resized_image = self.resize_with_input_factor(image)
        if self.image_need_tilling(resized_image):
            upscaled_image = self.AI_upscale_with_tilling(resized_image)
        else:
            upscaled_image = self.AI_upscale(resized_image)
        return self.resize_with_output_factor(upscaled_image)
