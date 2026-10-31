from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from qualityscaler.ai import AI_upscale, calculate_optimal_workers


class FakeInferenceSession:
    """Repeat NCHW pixels to simulate an upscaler without loading a model."""

    def __init__(self, factor):
        self.factor = factor
        self.inputs = []

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, output_names, inputs):
        image = inputs["input"]
        self.inputs.append(image.copy())
        return [
            np.repeat(
                np.repeat(image, self.factor, axis=2), self.factor, axis=3
            )
        ]


def make_upscaler(
    model="BSRGANx2", input_factor=1, output_factor=1, max_resolution=64
):
    upscaler = AI_upscale.__new__(AI_upscale)
    upscaler.AI_model_name = model
    upscaler.input_resize_factor = input_factor
    upscaler.output_resize_factor = output_factor
    upscaler.max_resolution = max_resolution
    upscaler.upscale_factor = upscaler._get_upscale_factor()
    upscaler.inferenceSession = FakeInferenceSession(upscaler.upscale_factor)
    return upscaler


class ImportAndSessionTests(unittest.TestCase):
    def test_backend_imports_do_not_require_gui_or_onnxruntime(self):
        script = """
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'customtkinter', 'tkinter', 'onnxruntime', 'QualityScaler'}:
        raise AssertionError('Unexpected backend dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import qualityscaler.ai
import qualityscaler.media
import qualityscaler.paths
assert 'onnxruntime' not in sys.modules
assert 'customtkinter' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_constructor_loads_session_lazily_with_directml_options(self):
        for gpu, options in (
            ("Auto", {"performance_preference": "high_performance"}),
            ("GPU 1", {"device_id": "0"}),
            ("GPU 2", {"device_id": "1"}),
            ("GPU 3", {"device_id": "2"}),
            ("GPU 4", {"device_id": "3"}),
        ):
            with self.subTest(gpu=gpu):
                fake_session = FakeInferenceSession(2)
                constructor = Mock(return_value=fake_session)
                session_options = SimpleNamespace(enable_mem_pattern=True,
                                                  execution_mode="parallel")
                with patch.dict(
                    sys.modules,
                    {"onnxruntime": SimpleNamespace(
                        InferenceSession=constructor,
                        SessionOptions=Mock(return_value=session_options),
                        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
                    )},
                ):
                    upscaler = AI_upscale("BSRGANx2", gpu, 0.5, 1, 512)
                expected_model = (
                    Path(__file__).resolve().parents[1]
                    / "AI-onnx" / "BSRGANx2_fp16.onnx"
                )
                self.assertEqual(upscaler.AI_model_path, str(expected_model))
                self.assertIs(upscaler.inferenceSession, fake_session)
                self.assertFalse(session_options.enable_mem_pattern)
                self.assertEqual(session_options.execution_mode, "sequential")
                constructor.assert_called_once_with(
                    path_or_bytes=str(expected_model),
                    sess_options=session_options,
                    providers=["DmlExecutionProvider"],
                    provider_options=[options],
                )


class SizingTests(unittest.TestCase):
    def test_model_upscale_factors(self):
        for model, factor in (
            ("IRCNN_Mx1", 1), ("BSRGANx2", 2), ("RealESR_Gx4", 4)
        ):
            with self.subTest(model=model):
                upscaler = make_upscaler(model=model)
                self.assertEqual(upscaler.upscale_factor, factor)
                self.assertEqual(
                    upscaler.calculate_target_resolution(np.zeros((3, 5, 3))),
                    (3 * factor, 5 * factor),
                )

    def test_resize_factors_use_even_dimensions_and_existing_interpolation(self):
        image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        for method_name, factor_name in (
            ("resize_with_input_factor", "input_resize_factor"),
            ("resize_with_output_factor", "output_resize_factor"),
        ):
            for factor, dimensions, interpolation in (
                (0.5, (4, 2), cv2.INTER_AREA),
                (1.5, (10, 8), cv2.INTER_CUBIC),
                (2, (14, 10), cv2.INTER_CUBIC),
            ):
                with self.subTest(method=method_name, factor=factor):
                    upscaler = make_upscaler()
                    setattr(upscaler, factor_name, factor)
                    np.testing.assert_array_equal(
                        getattr(upscaler, method_name)(image),
                        cv2.resize(image, dimensions, interpolation=interpolation),
                    )

    def test_unit_resize_factor_preserves_array_identity_and_odd_dimensions(self):
        upscaler = make_upscaler()
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        self.assertIs(upscaler.resize_with_input_factor(image), image)
        self.assertIs(upscaler.resize_with_output_factor(image), image)

    def test_thread_count_uses_decoded_frame_and_input_resize(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        for factor, max_resolution, selected, model, expected in (
            (1, 400, 6, "BSRGANx2", 6),
            (0.5, 400, 8, "BSRGANx2", 8),
            (2, 400, 8, "BSRGANx2", 2),
            (1, 50, 8, "BSRGANx2", 1),
            (1, 400, 0, "BSRGANx2", 1),
            (1, 400, 8, "IRCNN_Mx1", 1),
        ):
            with self.subTest(factor=factor, max=max_resolution, model=model):
                upscaler = make_upscaler(
                    model=model, input_factor=factor, max_resolution=max_resolution
                )
                with redirect_stdout(io.StringIO()):
                    actual = upscaler.calculate_optimal_multithreads_number(
                        frame, selected
                    )
                self.assertEqual(actual, expected)

    def test_tiling_threshold_matches_pixel_capacity(self):
        upscaler = make_upscaler(max_resolution=8)
        self.assertFalse(upscaler.image_need_tilling(np.zeros((4, 16, 3))))
        self.assertTrue(upscaler.image_need_tilling(np.zeros((8, 12, 3))))
        self.assertEqual(upscaler.calculate_tiles_number(np.zeros((8, 12, 3))), (2, 1))

    def test_worker_sizing_never_loads_a_model_or_allocates_a_resized_image(self):
        frame = np.zeros((5, 7, 3), dtype=np.uint8)
        with patch.object(AI_upscale, "_load_inferenceSession",
                          side_effect=AssertionError("Worker sizing loaded a model")), \
                patch("qualityscaler.ai.cv2.resize",
                      side_effect=AssertionError("Worker sizing resized the image")):
            with redirect_stdout(io.StringIO()):
                # Unit factors preserve odd dimensions; 2x resizing rounds even.
                self.assertEqual(calculate_optimal_workers(frame, "Testx2", 1, 10, 8), 2)
                self.assertEqual(calculate_optimal_workers(frame, "Testx2", 2, 20, 8), 2)
            with self.assertRaisesRegex(ValueError, "empty frame"):
                calculate_optimal_workers(frame, "Testx2", 0.01, 20, 8)


class InferenceTests(unittest.TestCase):
    def test_preprocess_postprocess_and_clipping(self):
        upscaler = make_upscaler()
        image = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 18
        tensor = upscaler.preprocess_image(image)
        self.assertEqual(tensor.shape, (1, 3, 2, 3))
        np.testing.assert_array_equal(upscaler.postprocess_output(tensor), image)
        np.testing.assert_array_equal(
            upscaler.postprocess_output(np.array([[[[-1, 2]]] * 3], dtype=np.float32)),
            np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32),
        )

    def test_rgb_grayscale_and_alpha_inference_without_a_model(self):
        for channels in (None, 3, 4):
            with self.subTest(channels=channels):
                shape = (4, 6) if channels is None else (4, 6, channels)
                image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
                upscaler = make_upscaler()
                result = upscaler.AI_orchestration(image)
                expected = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
                np.testing.assert_allclose(result, expected, atol=1)
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(
                    len(upscaler.inferenceSession.inputs), 2 if channels == 4 else 1
                )
                for tensor in upscaler.inferenceSession.inputs:
                    self.assertEqual(tensor.shape, (1, 3, 4, 6))
                    self.assertEqual(tensor.dtype, np.float32)

    def test_orchestration_applies_input_and_output_resizes(self):
        upscaler = make_upscaler(input_factor=0.5, output_factor=0.5)
        image = np.full((8, 12, 3), 128, dtype=np.uint8)
        result = upscaler.AI_orchestration(image)
        self.assertEqual(upscaler.inferenceSession.inputs[0].shape, (1, 3, 4, 6))
        self.assertEqual(result.shape, (4, 6, 3))
        np.testing.assert_allclose(result, 128, atol=1)

    def test_orchestration_recombines_rgb_grayscale_and_rgba_tiles(self):
        for channels in (None, 3, 4):
            with self.subTest(channels=channels):
                shape = (8, 12) if channels is None else (8, 12, channels)
                image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
                upscaler = make_upscaler(max_resolution=8)
                result = upscaler.AI_orchestration(image)
                expected = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
                np.testing.assert_allclose(result, expected, atol=1)
                self.assertEqual(result.shape, expected.shape)
                self.assertEqual(
                    len(upscaler.inferenceSession.inputs), 4 if channels == 4 else 2
                )

    def test_existing_high_bit_depth_normalization_is_retained(self):
        upscaler = make_upscaler()
        image = np.full((2, 3, 3), 65535, dtype=np.uint16)
        result = upscaler.AI_upscale(image)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (4, 6, 3))
        np.testing.assert_array_equal(result, np.full((4, 6, 3), 65535, np.float32))


if __name__ == "__main__":
    unittest.main()
