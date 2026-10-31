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

from qualityscaler.ai import (
    AI_upscale, calculate_optimal_workers, get_model_upscale_factor, scaled_dimensions,
)


class FakeInferenceSession:
    """Repeat NCHW pixels to simulate an upscaler without loading a model."""

    def __init__(self, factor):
        self.factor = factor
        self.inputs = []
        self.input_dicts = []
        self.metadata_reads = 0

    def get_inputs(self):
        self.metadata_reads += 1
        return [SimpleNamespace(name="input")]

    def run(self, output_names, inputs):
        image = inputs["input"]
        if not image.flags.c_contiguous:
            raise AssertionError("Inference requires contiguous NCHW input")
        self.inputs.append(image.copy())
        self.input_dicts.append(inputs)
        return [
            np.repeat(
                np.repeat(image, self.factor, axis=2), self.factor, axis=3
            )
        ]


def make_upscaler(model="BSRGANx2", input_factor=1, tiles_resolution=64):
    upscaler = AI_upscale(model, "Auto", input_factor, tiles_resolution)
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
import qualityscaler.gpu
import qualityscaler.media
import qualityscaler.paths
from qualityscaler.ai import AI_upscale, calculate_optimal_workers
import numpy as np
engine = AI_upscale('BSRGANx2', 'Auto', 1, 512)
assert engine.inferenceSession is None
assert calculate_optimal_workers(np.zeros((5, 7, 3)), 'IRCNN_Mx1', 1, 10, 8) == 2
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

    def test_constructor_is_lazy_and_ensure_session_configures_directml(self):
        for gpu, options in (
            ("Auto", {"performance_preference": "high_performance"}),
            ("0", {"device_id": "0"}),
            ("2", {"device_id": "2"}),
            ("GPU1", {"device_id": "0"}),
            ("GPU 1", {"device_id": "0"}),
            ("GPU 2", {"device_id": "1"}),
        ):
            with self.subTest(gpu=gpu):
                fake_session = FakeInferenceSession(2)
                constructor = Mock(return_value=fake_session)
                session_options = SimpleNamespace(
                    enable_mem_pattern=True, execution_mode="parallel",
                )
                with patch.dict(
                    sys.modules,
                    {"onnxruntime": SimpleNamespace(
                        InferenceSession=constructor,
                        SessionOptions=Mock(return_value=session_options),
                        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
                        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
                        get_available_providers=lambda: ["DmlExecutionProvider"],
                    )},
                ):
                    upscaler = AI_upscale(
                        selected_AI_model="BSRGANx2", selected_gpu=gpu,
                        input_resize_factor=0.5, tiles_resolution=512,
                    )
                    constructor.assert_not_called()
                    self.assertIsNone(upscaler.inferenceSession)
                    self.assertIsNone(upscaler.input_name)
                    self.assertIsNone(upscaler.onnx_input)
                    self.assertIs(upscaler.ensure_session(), fake_session)
                    input_dict = upscaler.onnx_input
                    self.assertIs(upscaler.ensure_session(), fake_session)
                    self.assertIs(upscaler.onnx_input, input_dict)
                expected_model = (
                    Path(__file__).resolve().parents[1]
                    / "AI-onnx" / "BSRGANx2_fp16.onnx"
                )
                self.assertEqual(upscaler.selected_AI_model_path, str(expected_model))
                self.assertIs(upscaler.inferenceSession, fake_session)
                self.assertEqual(upscaler.input_name, "input")
                self.assertEqual(upscaler.onnx_input, {"input": None})
                self.assertEqual(fake_session.metadata_reads, 1)
                self.assertFalse(session_options.enable_mem_pattern)
                self.assertEqual(session_options.execution_mode, "sequential")
                self.assertFalse(session_options.enable_profiling)
                self.assertEqual(session_options.intra_op_num_threads, 1)
                self.assertEqual(session_options.graph_optimization_level, "all")
                constructor.assert_called_once_with(
                    path_or_bytes=str(expected_model),
                    sess_options=session_options,
                    providers=["DmlExecutionProvider"],
                    provider_options=[options],
                )

    def test_cpu_fallback_has_aligned_empty_provider_options(self):
        session_options = SimpleNamespace(enable_mem_pattern=True, execution_mode="default")
        constructor = Mock(return_value=FakeInferenceSession(2))
        with patch.dict(sys.modules, {"onnxruntime": SimpleNamespace(
            InferenceSession=constructor,
            SessionOptions=lambda: session_options,
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
            get_available_providers=lambda: ["CPUExecutionProvider"],
        )}):
            engine = AI_upscale("BSRGANx2", "GPU1", 1, 512)
            engine.ensure_session()
        self.assertEqual(constructor.call_args.kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(constructor.call_args.kwargs["provider_options"], [{}])
        self.assertEqual(session_options.intra_op_num_threads, 1)
        self.assertEqual(session_options.graph_optimization_level, "all")
        self.assertTrue(session_options.enable_mem_pattern)
        self.assertEqual(session_options.execution_mode, "default")

    def test_failed_session_load_can_be_retried(self):
        engine = AI_upscale("BSRGANx2", "Auto", 1, 512)
        session = FakeInferenceSession(2)
        with patch.object(engine, "_load_inferenceSession",
                          side_effect=[RuntimeError("probe failure"), session]) as loader:
            with self.assertRaisesRegex(RuntimeError, "probe failure"):
                engine.ensure_session()
            self.assertIsNone(engine.inferenceSession)
            self.assertIs(engine.ensure_session(), session)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(engine.onnx_input, {"input": None})

    def test_orchestration_initializes_once_and_reuses_cached_input_metadata(self):
        engine = AI_upscale("BSRGANx2", "Auto", 1, 512)
        session = FakeInferenceSession(2)
        with patch.object(engine, "_load_inferenceSession", return_value=session) as loader:
            engine.AI_orchestration(np.full((4, 6, 3), 40, np.uint8))
            engine.AI_orchestration(np.full((4, 6, 3), 90, np.uint8))
        loader.assert_called_once_with()
        self.assertEqual(session.metadata_reads, 1)
        self.assertIs(session.input_dicts[0], session.input_dicts[1])


class SizingTests(unittest.TestCase):
    def test_model_upscale_factors(self):
        for model, factor in (
            ("IRCNN_Mx1", 1), ("BSRGANx2", 2), ("Testx3", 3),
            ("RealESR_Gx4", 4), ("Unknown", 1),
        ):
            with self.subTest(model=model):
                upscaler = make_upscaler(model=model)
                self.assertEqual(upscaler.upscale_factor, factor)
                self.assertEqual(get_model_upscale_factor(model), factor)
                self.assertEqual(
                    upscaler.calculate_target_resolution(np.zeros((3, 5, 3))),
                    (3 * factor, 5 * factor),
                )

    def test_input_resize_uses_shared_even_dimensions_and_upstream_interpolation(self):
        image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        for factor, dimensions in (
            (0.5, (4, 2)), (1.5, (10, 8)), (2, (14, 10)),
        ):
            with self.subTest(factor=factor):
                upscaler = make_upscaler(input_factor=factor)
                self.assertEqual(scaled_dimensions(image, factor), dimensions)
                np.testing.assert_array_equal(
                    upscaler.resize_with_input_factor(image),
                    cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA),
                )

    def test_unit_resize_factor_still_rounds_odd_dimensions_even_like_2026(self):
        upscaler = make_upscaler()
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        self.assertEqual(scaled_dimensions(image, 1), (8, 6))
        result = upscaler.resize_with_input_factor(image)
        self.assertEqual(result.shape, (6, 8, 3))
        np.testing.assert_array_equal(
            result, cv2.resize(image, (8, 6), interpolation=cv2.INTER_AREA),
        )
        self.assertFalse(hasattr(upscaler, "resize_with_output_factor"))

    def test_thread_count_uses_decoded_frame_and_input_resize(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        for factor, max_resolution, selected, model, expected in (
            (1, 400, 6, "BSRGANx2", 6),
            (0.5, 400, 8, "BSRGANx2", 8),
            (2, 400, 8, "BSRGANx2", 2),
            (1, 50, 8, "BSRGANx2", 1),
            (1, 400, 0, "BSRGANx2", 1),
            (1, 400, 8, "IRCNN_Mx1", 8),
        ):
            with self.subTest(factor=factor, max=max_resolution, model=model):
                upscaler = make_upscaler(
                    model=model, input_factor=factor, tiles_resolution=max_resolution
                )
                with redirect_stdout(io.StringIO()):
                    actual = upscaler.calculate_optimal_multithreads_number(
                        frame, selected
                    )
                self.assertEqual(actual, expected)

    def test_tiling_threshold_matches_pixel_capacity(self):
        upscaler = make_upscaler(tiles_resolution=8)
        self.assertFalse(upscaler.image_need_tilling(np.zeros((4, 16, 3))))
        self.assertTrue(upscaler.image_need_tilling(np.zeros((8, 12, 3))))
        self.assertEqual(upscaler.calculate_tiles_number(np.zeros((8, 12, 3))), (2, 1))

    def test_worker_sizing_never_loads_a_model_or_allocates_a_resized_image(self):
        frame = np.zeros((5, 7, 3), dtype=np.uint8)
        with patch("qualityscaler.ai.AI_upscale",
                   side_effect=AssertionError("Worker sizing constructed an engine")), \
                patch("qualityscaler.ai.cv2.resize",
                      side_effect=AssertionError("Worker sizing resized the image")):
            with redirect_stdout(io.StringIO()):
                # Every factor, including 1x, uses the media even-dimension rule.
                self.assertEqual(calculate_optimal_workers(frame, "Testx2", 1, 9, 8), 1)
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
        self.assertTrue(tensor.flags.c_contiguous)
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

    def test_orchestration_only_resizes_input_and_returns_native_model_scale(self):
        upscaler = make_upscaler(model="Testx3", input_factor=0.5)
        image = np.full((8, 12, 3), 128, dtype=np.uint8)
        result = upscaler.AI_orchestration(image)
        self.assertEqual(upscaler.inferenceSession.inputs[0].shape, (1, 3, 4, 6))
        self.assertEqual(result.shape, (12, 18, 3))
        np.testing.assert_allclose(result, 128, atol=1)

    def test_2026_high_bit_depth_normalization_returns_float_without_rounding(self):
        upscaler = make_upscaler()
        image = np.full((2, 3, 3), 65535, dtype=np.uint16)
        result = upscaler.AI_upscale(image)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (4, 6, 3))
        np.testing.assert_array_equal(result, np.full((4, 6, 3), 65535, np.float32))
        result = upscaler.de_normalize_image(np.full((2, 3, 3), 0.12345, np.float32), 65535)
        np.testing.assert_array_equal(result, np.full((2, 3, 3), 0.12345 * 65535, np.float32))
        self.assertNotEqual(result[0, 0, 0], round(float(result[0, 0, 0])))

    def test_uint8_normalization_skips_max_scan(self):
        upscaler = make_upscaler()
        with patch("qualityscaler.ai.np.max", side_effect=AssertionError("uint8 max scan")):
            image, max_range = upscaler.normalize_image(np.full((2, 3, 3), 255, np.float32), np.uint8)
        self.assertEqual(max_range, 255)
        np.testing.assert_array_equal(image, 1)

    def test_non_uint8_normalization_uses_2026_max_range_boundary(self):
        upscaler = make_upscaler()
        for max_value, expected_range in ((256, 255), (257, 65535)):
            with self.subTest(max_value=max_value):
                image = np.full((2, 3, 3), max_value, np.float32)
                normalized, actual_range = upscaler.normalize_image(image, np.uint16)
                self.assertEqual(actual_range, expected_range)
                np.testing.assert_allclose(normalized, max_value / expected_range)

    def test_float32_source_is_not_mutated_by_normalization(self):
        image = np.full((3, 5, 3), 128, np.float32)
        make_upscaler().AI_orchestration(image)
        np.testing.assert_array_equal(image, 128)


class FeatherTilingTests(unittest.TestCase):
    def test_feather_weights_are_positive_symmetric_and_cap_overlapping_ramps(self):
        engine = make_upscaler()
        for length, ramp, expected in (
            (7, 2, [1 / 3, 2 / 3, 1, 1, 1, 2 / 3, 1 / 3]),
            (5, 16, [1 / 3, 2 / 3, 1, 2 / 3, 1 / 3]),
            (4, 16, [1 / 3, 2 / 3, 2 / 3, 1 / 3]),
            (1, 16, [1]), (3, 0, [1, 1, 1]),
        ):
            with self.subTest(length=length, ramp=ramp):
                actual = engine._tile_feather_weights(length, ramp)
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_allclose(actual, expected)
                np.testing.assert_array_equal(actual, actual[::-1])
                self.assertTrue(np.all(actual > 0))

    def test_uneven_remainders_preserve_every_pixel_for_all_modes_and_factors(self):
        rng = np.random.default_rng(7)
        # Both axes have remainders. Skinny images exercise one-pixel reflect
        # padding and a single row/column of tiles.
        for height, width, limit in ((23, 29, 8), (1, 71, 8), (73, 1, 8)):
            for channels in (None, 3, 4):
                shape = (height, width) if channels is None else (height, width, channels)
                image = rng.integers(16, 240, shape, dtype=np.uint8)
                for factor in (1, 2, 3, 4):
                    with self.subTest(shape=shape, factor=factor):
                        engine = make_upscaler(model=f"Testx{factor}", tiles_resolution=limit)
                        result = engine.AI_upscale_with_tilling(image)
                        expected = np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)
                        self.assertEqual(result.shape, expected.shape)
                        self.assertEqual(result.dtype, np.uint8)
                        np.testing.assert_allclose(result, expected, atol=1)
                        # Explicit edge assertions catch dropped last-row/column
                        # remainders even when the rest of the image blends well.
                        np.testing.assert_allclose(result[-factor:], expected[-factor:], atol=1)
                        np.testing.assert_allclose(result[:, -factor:], expected[:, -factor:], atol=1)
                        tiles_x, tiles_y = engine.calculate_tiles_number(image)
                        self.assertEqual(
                            len(engine.inferenceSession.inputs),
                            tiles_x * tiles_y * (2 if channels == 4 else 1),
                        )

    def test_tiles_include_exact_reflect_context_and_all_uneven_remainders(self):
        image = np.arange(23 * 29 * 3, dtype=np.uint16).reshape(23, 29, 3).astype(np.uint8)
        engine = make_upscaler(tiles_resolution=8)
        padded = np.pad(image, ((16, 16), (16, 16), (0, 0)), mode="reflect")
        seen = []

        def upscale(tile):
            seen.append(tile.copy())
            return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)

        with patch.object(engine, "AI_upscale", side_effect=upscale):
            result = engine.AI_upscale_with_tilling(image)
        # 4 x 3 cores, floor-sized until the last row/column: widths 7/8,
        # heights 7/9. Each receives 16 context pixels on each side.
        self.assertEqual(len(seen), 12)
        np.testing.assert_array_equal(seen[0], padded[:39, :39])
        np.testing.assert_array_equal(seen[-1], padded[14:55, 21:61])
        self.assertEqual(seen[-1].shape, (41, 40, 3))
        np.testing.assert_array_equal(result, np.repeat(np.repeat(image, 2, axis=0), 2, axis=1))

    def test_blend_uses_normalized_feather_weights_not_crop_overwrite_or_average(self):
        # Two unequal cores: width 32 and 33, 16px context each side.
        # Their constant outputs make the expected feather contribution analytic.
        image = np.zeros((8, 65, 3), np.uint8)
        engine = make_upscaler(model="Testx3", tiles_resolution=40)
        factor = 3
        ramp = 16 * factor
        first_width = (32 + 32) * factor
        second_width = (33 + 32) * factor
        calls = []

        def upscale(tile):
            value = 40 if not calls else 200
            calls.append(tile.shape)
            return np.full((tile.shape[0] * factor, tile.shape[1] * factor, 3), value, np.uint8)

        with patch.object(engine, "AI_upscale", side_effect=upscale):
            result = engine.AI_upscale_with_tilling(image)
        self.assertEqual(calls, [(40, 64, 3), (40, 65, 3)])
        for x in (0, 47, 48, 49, 73, 95, 96, 120, 143, 144, 194):
            # Evaluate each tile's 1-D weight directly, independently of the
            # production feather helper. Vertical weights cancel in this case.
            def weight(position, length):
                if not 0 <= position < length:
                    return 0
                edge_distance = min(position + 1, length - position)
                return min(1, edge_distance / (ramp + 1))

            first_weight = weight(x + ramp, first_width)
            second_weight = weight(x + ramp - 32 * factor, second_width)
            expected = int((40 * first_weight + 200 * second_weight)
                           / (first_weight + second_weight) + 0.5)
            np.testing.assert_array_equal(result[:, x], expected)
        self.assertTrue(np.all(np.diff(result[0, :, 0].astype(int)) >= 0))
        self.assertLess(np.max(np.diff(result[0, :, 0].astype(int))), 6)

    def test_four_tile_intersection_uses_separable_weights_in_both_axes(self):
        image = np.zeros((65, 67), np.uint8)
        engine = make_upscaler(model="Testx3", tiles_resolution=40)
        values = [20, 70, 150, 230]
        calls = []

        def upscale(tile):
            value = values[len(calls)]
            calls.append(tile.shape)
            return np.full((tile.shape[0] * 3, tile.shape[1] * 3), value, np.uint8)

        with patch.object(engine, "AI_upscale", side_effect=upscale):
            result = engine.AI_upscale_with_tilling(image)
        self.assertEqual(result.shape, (195, 201))
        self.assertEqual(calls, [(64, 65), (64, 66), (65, 65), (65, 66)])

        def weight(position, length):
            if not 0 <= position < length:
                return 0
            return min(1, min(position + 1, length - position) / 49)

        for y, x in ((0, 0), (194, 200), (73, 80), (95, 99), (121, 130), (140, 60)):
            numerator = denominator = 0
            for index, (core_y, core_x) in enumerate(((0, 0), (0, 33), (32, 0), (32, 33))):
                tile_height, tile_width = calls[index]
                contribution = (
                    weight(y + 48 - core_y * 3, tile_height * 3)
                    * weight(x + 48 - core_x * 3, tile_width * 3)
                )
                numerator += values[index] * contribution
                denominator += contribution
            self.assertAlmostEqual(int(result[y, x]), int(numerator / denominator + 0.5), delta=1)

    def test_float_tiles_do_not_renormalize_shared_overlap_context(self):
        image = np.full((23, 29, 3), 128, np.float32)
        engine = make_upscaler(tiles_resolution=8)
        np.testing.assert_allclose(engine.AI_orchestration(image), 128, atol=1)
        np.testing.assert_array_equal(image, 128)


if __name__ == "__main__":
    unittest.main()
