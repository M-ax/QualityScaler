"""Single-frame backend checks without models, a GPU, Tk, or output caches."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from qualityscaler import preview
from qualityscaler.config import ProcessingConfig
from qualityscaler.media import image_write, resize_with_output_factor, sharpen_image


def make_config(**changes):
    config = ProcessingConfig(
        selected_file_list=[], selected_output_path="unused-output",
        selected_AI_model="Testx2", selected_AI_multithreading=4,
        input_resize_factor=0.5, output_resize_factor=1.5,
        selected_gpu="2", tiles_resolution=512, selected_sharpening_amount=0.3,
        selected_keep_frames=True, selected_image_extension=".jpg",
        selected_video_extension=".mp4", selected_video_codec="x265",
    )
    return replace(config, **changes)


def make_capture(*, count=4, fps=24.0):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = {
        cv2.CAP_PROP_FRAME_COUNT: count, cv2.CAP_PROP_FPS: fps,
    }.__getitem__
    capture.set.return_value = True
    capture.read.return_value = (True, np.full((8, 12, 3), 90, np.uint8))
    return capture


class FakeInferenceSession:
    """Repeat native-scale tensor pixels, leaving real orchestration in use."""

    def __init__(self):
        self.inputs = []

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, output_names, inputs):
        self.inputs.append(inputs["input"].copy())
        return [np.repeat(np.repeat(inputs["input"], 2, axis=2), 2, axis=3)]


class ProbeTests(unittest.TestCase):
    def test_metadata_is_frozen_and_spawn_serializable(self):
        source = preview.PreviewSource("input.png", 1, 30.0, False)
        with self.assertRaises(FrozenInstanceError):
            source.frame_count = 7
        self.assertEqual(pickle.loads(pickle.dumps(source)), source)

    def test_still_metadata_uses_real_unicode_image_decoder_not_capture(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "画像 🐈.PNG"
            image_write(str(path), np.zeros((5, 7, 4), np.uint16), ".png")
            with patch.object(preview.cv2, "VideoCapture") as capture:
                source = preview.probe_source(str(path))
            self.assertEqual(source, preview.PreviewSource(str(path), 1, 30.0, False))
            capture.assert_not_called()

    def test_corrupt_and_missing_still_images_raise(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            with self.assertRaises(FileNotFoundError):
                preview.probe_source(str(path))
            for data in (b"", b"not an image"):
                with self.subTest(data=data):
                    path.write_bytes(data)
                    with self.assertRaisesRegex(ValueError, "Unable to decode image"):
                        preview.probe_source(str(path))

    def test_video_metadata_checks_one_decoded_frame_and_releases(self):
        for path in ("input.MP4", "animated.GIF", "old.WMV"):
            with self.subTest(path=path):
                capture = make_capture(count=7, fps=29.97)
                with patch.object(preview.cv2, "VideoCapture", return_value=capture):
                    source = preview.probe_source(path)
                self.assertEqual(source, preview.PreviewSource(path, 7, 29.97, True))
                capture.read.assert_called_once_with()
                capture.set.assert_not_called()
                capture.release.assert_called_once_with()

    def test_invalid_video_fps_uses_shared_sanitizer(self):
        for fps in (0, -1, float("inf"), float("nan"), None):
            with self.subTest(fps=fps):
                capture = make_capture(fps=fps)
                with patch.object(preview.cv2, "VideoCapture", return_value=capture):
                    self.assertEqual(preview.probe_source("input.mp4").fps, 30.0)
                capture.release.assert_called_once_with()

    def test_unknown_frame_count_is_rejected_without_scanning(self):
        for count in (0, -1, 0.5, float("inf"), float("-inf"), float("nan")):
            with self.subTest(count=count):
                capture = make_capture(count=count)
                with patch.object(preview.cv2, "VideoCapture", return_value=capture):
                    with self.assertRaisesRegex(ValueError, "video frame count"):
                        preview.probe_source("input.mp4")
                capture.read.assert_not_called()
                capture.release.assert_called_once_with()

    def test_closed_or_undecodable_video_is_rejected_and_released(self):
        for opened, decoded in (
            (False, (True, np.zeros((2, 2, 3), np.uint8))),
            (True, (False, None)),
            (True, (True, None)),
            (True, (True, np.empty((0, 0, 3), np.uint8))),
        ):
            with self.subTest(opened=opened, decoded=decoded):
                capture = make_capture()
                capture.isOpened.return_value = opened
                capture.read.return_value = decoded
                with patch.object(preview.cv2, "VideoCapture", return_value=capture):
                    with self.assertRaisesRegex(ValueError, "Unable to (open|read) video"):
                        preview.probe_source("input.mp4")
                capture.release.assert_called_once_with()

    def test_capture_released_when_probe_operation_raises(self):
        for operation in ("isOpened", "get", "read"):
            with self.subTest(operation=operation):
                capture = make_capture()
                getattr(capture, operation).side_effect = RuntimeError("decoder failed")
                with patch.object(preview.cv2, "VideoCapture", return_value=capture):
                    with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                        preview.probe_source("input.mp4")
                capture.release.assert_called_once_with()


class EngineAndRenderTests(unittest.TestCase):
    def test_engine_constructor_uses_only_session_choices(self):
        config = make_config()
        with patch.object(preview, "AI_upscale") as constructor:
            engine = preview.create_preview_engine(config)
        self.assertIs(engine, constructor.return_value)
        constructor.assert_called_once_with("Testx2", "2", 0.5, 512)

    def test_image_resizes_before_sharpening_and_can_create_engine(self):
        config = make_config()
        source = preview.PreviewSource("photo.png", 1, 30.0, False)
        original = np.zeros((8, 12, 3), np.uint8)
        native = np.zeros((8, 12, 3), np.uint8)
        resized = np.zeros((12, 18, 3), np.uint8)
        sharpened = resized + 1
        events = []
        engine = Mock()
        engine.AI_orchestration.side_effect = lambda image: events.append("AI") or native
        with patch.object(preview, "image_read", return_value=original), \
                patch.object(preview, "create_preview_engine", return_value=engine) as create, \
                patch.object(preview, "resize_with_output_factor",
                             side_effect=lambda *args: events.append("resize") or resized) as resize, \
                patch.object(preview, "sharpen_image",
                             side_effect=lambda *args: events.append("sharpen") or sharpened) as sharpen:
            actual = preview.render_preview(source, 999, config)
        self.assertIs(actual, sharpened)
        self.assertEqual(events, ["AI", "resize", "sharpen"])
        create.assert_called_once_with(config)
        engine.AI_orchestration.assert_called_once_with(original)
        resize.assert_called_once_with(native, 1.5)
        sharpen.assert_called_once_with(resized, 0.3)

    def test_video_seeks_to_zero_based_clamped_index_and_reads_once(self):
        source = preview.PreviewSource("input.mp4", 4, 24.0, True)
        config = make_config(selected_sharpening_amount=0)
        engine = Mock()
        engine.AI_orchestration.return_value = np.zeros((8, 12, 3), np.uint8)
        for request, expected in ((-10, 0), (0, 0), (1, 1), (3, 3), (4, 3), (999, 3)):
            with self.subTest(request=request):
                capture = make_capture()
                with patch.object(preview.cv2, "VideoCapture", return_value=capture), \
                        patch.object(preview, "create_preview_engine") as create, \
                        patch.object(preview, "image_read") as image_read:
                    preview.render_preview(source, request, config, engine)
                capture.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, expected)
                capture.read.assert_called_once_with()
                capture.release.assert_called_once_with()
                create.assert_not_called()
                image_read.assert_not_called()

    def test_video_sharpens_before_resizing_to_production_geometry(self):
        config = make_config(input_resize_factor=0.53, output_resize_factor=1)
        source = preview.PreviewSource("input.mp4", 4, 24.0, True)
        capture = make_capture()
        original = np.zeros((100, 100, 3), np.uint8)
        capture.read.return_value = (True, original)
        inference_input = np.zeros((54, 54, 3), np.uint8)
        native = np.zeros((108, 108, 3), np.uint8)
        sharpened = native + 1
        resized = np.zeros((104, 104, 3), np.uint8)
        events = []
        engine = Mock()
        engine.AI_orchestration.side_effect = lambda *args, **kwargs: events.append("AI") or native
        capture.release.side_effect = lambda: events.append("release")

        def resize_frame(image, size, *, interpolation):
            if interpolation == cv2.INTER_AREA:
                events.append("resize_input")
                return inference_input
            events.append("resize_output")
            return resized

        with patch.object(preview.cv2, "VideoCapture", return_value=capture), \
                patch.object(preview, "sharpen_image",
                             side_effect=lambda *args: events.append("sharpen") or sharpened) as sharpen, \
                patch.object(preview.cv2, "resize", side_effect=resize_frame) as resize:
            actual = preview.render_preview(source, 2, config, engine)
        self.assertIs(actual, resized)
        self.assertEqual(events, ["release", "resize_input", "AI", "sharpen", "resize_output"])
        engine.AI_orchestration.assert_called_once_with(inference_input, input_is_scaled=True)
        sharpen.assert_called_once_with(native, 0.3)
        # Geometry is based on the original video, not rounded inference pixels.
        self.assertEqual(resize.call_count, 2)
        self.assertIs(resize.call_args_list[0].args[0], original)
        self.assertEqual(resize.call_args_list[0].args[1], (54, 54))
        self.assertEqual(resize.call_args_list[0].kwargs, {"interpolation": cv2.INTER_AREA})
        self.assertIs(resize.call_args.args[0], sharpened)
        self.assertEqual(resize.call_args.args[1], (104, 104))
        self.assertEqual(resize.call_args.kwargs, {"interpolation": cv2.INTER_CUBIC})

    def test_zero_sharpening_skips_helper_for_images_and_videos(self):
        config = make_config(selected_sharpening_amount=0)
        engine = Mock()
        engine.AI_orchestration.return_value = np.zeros((8, 12, 3), np.uint8)
        for is_video in (False, True):
            with self.subTest(is_video=is_video):
                source = preview.PreviewSource("input", 1, 30.0, is_video)
                with patch.object(preview, "image_read", return_value=np.zeros((8, 12, 3), np.uint8)), \
                        patch.object(preview.cv2, "VideoCapture", return_value=make_capture()), \
                        patch.object(preview, "sharpen_image") as sharpen:
                    preview.render_preview(source, 0, config, engine)
                sharpen.assert_not_called()

    def test_failed_video_open_seek_or_decode_releases_without_inference(self):
        source = preview.PreviewSource("input.mp4", 4, 24.0, True)
        for operation in ("open", "seek", "read", "empty", "exception"):
            with self.subTest(operation=operation):
                capture = make_capture()
                if operation == "open":
                    capture.isOpened.return_value = False
                elif operation == "seek":
                    capture.set.return_value = False
                elif operation == "read":
                    capture.read.return_value = (False, None)
                elif operation == "empty":
                    capture.read.return_value = (True, np.empty((0, 0, 3), np.uint8))
                else:
                    capture.read.side_effect = RuntimeError("decoder failed")
                engine = Mock()
                with patch.object(preview.cv2, "VideoCapture", return_value=capture):
                    with self.assertRaises((ValueError, RuntimeError)):
                        preview.render_preview(source, 2, make_config(), engine)
                capture.release.assert_called_once_with()
                engine.AI_orchestration.assert_not_called()

    def test_inference_failure_propagates_after_releasing_video(self):
        source = preview.PreviewSource("input.mp4", 4, 24.0, True)
        capture = make_capture()
        engine = Mock()
        engine.AI_orchestration.side_effect = RuntimeError("inference failed")
        with patch.object(preview.cv2, "VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                preview.render_preview(source, 1, make_config(), engine)
        capture.release.assert_called_once_with()

    def test_invalid_source_count_is_rejected_before_io(self):
        with patch.object(preview.cv2, "VideoCapture") as capture:
            with self.assertRaisesRegex(ValueError, "at least one frame"):
                preview.render_preview(
                    preview.PreviewSource("input.mp4", 0, 30.0, True), 0, make_config()
                )
        capture.assert_not_called()


class RealMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_video_input_scales_once_and_clamps_tiny_axes_without_mutating_engine(self):
        for shape, factor, input_size, output_size in (
            ((100, 100, 3), 0.53, (54, 54), (104, 104)),
            ((3, 100, 3), 0.1, (10, 2), (20, 4)),
            ((5, 3, 3), 0.001, (2, 2), (4, 8)),
        ):
            with self.subTest(shape=shape, factor=factor):
                config = make_config(
                    input_resize_factor=factor, output_resize_factor=1,
                    selected_sharpening_amount=0,
                )
                original = np.random.default_rng(5).integers(16, 240, shape, np.uint8)
                capture = make_capture()
                capture.read.return_value = (True, original)
                session = FakeInferenceSession()
                engine = preview.create_preview_engine(config)
                engine.inferenceSession = session
                with patch.object(preview.cv2, "VideoCapture", return_value=capture), \
                        patch.object(engine, "resize_with_input_factor",
                                     side_effect=AssertionError("input scaled twice")):
                    actual = preview.render_preview(
                        preview.PreviewSource("input.mp4", 4, 24.0, True), 0, config, engine
                    )
                expected_input = cv2.resize(original, input_size, interpolation=cv2.INTER_AREA)
                self.assertEqual(session.inputs[0].shape, (1, 3, input_size[1], input_size[0]))
                np.testing.assert_allclose(
                    session.inputs[0][0].transpose(1, 2, 0), expected_input / 255, atol=1e-7
                )
                self.assertEqual(actual.shape[1::-1], output_size)
                self.assertEqual(engine.input_resize_factor, factor)
                self.assertIs(engine.inferenceSession, session)

    def test_stills_use_real_input_scaling_inference_and_output_helpers(self):
        for channels in (None, 3, 4):
            with self.subTest(channels=channels):
                shape = (8, 12) if channels is None else (8, 12, channels)
                image = np.random.default_rng(5).integers(16, 240, shape, np.uint8)
                path = self.root / f"画像 {channels}.png"
                image_write(str(path), image, ".png")
                config = make_config(selected_sharpening_amount=0)
                source = preview.probe_source(str(path))
                before = set(self.root.rglob("*"))
                with patch.object(preview.AI_upscale, "_load_inferenceSession",
                                  return_value=FakeInferenceSession()):
                    engine = preview.create_preview_engine(config)
                    actual = preview.render_preview(source, -1, config, engine)
                    # Reusing an engine can change output scale/sharpening only.
                    changed = replace(config, output_resize_factor=1, selected_sharpening_amount=0.5)
                    second = preview.render_preview(source, 99, changed, engine)
                input_image = cv2.resize(image, (6, 4), interpolation=cv2.INTER_AREA)
                native = np.repeat(np.repeat(input_image, 2, axis=0), 2, axis=1)
                expected = resize_with_output_factor(native, 1.5)
                np.testing.assert_allclose(actual, expected, atol=1)
                np.testing.assert_allclose(
                    second, sharpen_image(resize_with_output_factor(native, 1), 0.5), atol=2
                )
                self.assertEqual(actual.shape, (12, 18) + (() if channels is None else (channels,)))
                self.assertEqual(actual.dtype, np.uint8)
                self.assertEqual(set(self.root.rglob("*")), before, "preview wrote output or cache")

    def test_real_video_seek_and_clamping_with_reused_mocked_inference(self):
        path = self.root / "seek test.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (32, 24))
        try:
            if not writer.isOpened():
                self.skipTest("OpenCV MJPEG video writer is unavailable")
            for level in (32, 96, 160, 224):
                writer.write(np.full((24, 32, 3), (level, 20, 40), np.uint8))
        finally:
            writer.release()
        source = preview.probe_source(str(path))
        self.assertEqual(source.frame_count, 4)
        self.assertAlmostEqual(source.fps, 8)
        self.assertTrue(source.is_video)
        config = make_config(selected_sharpening_amount=0, output_resize_factor=0.5)
        before = set(self.root.rglob("*"))
        with patch.object(preview.AI_upscale, "_load_inferenceSession",
                          return_value=FakeInferenceSession()) as load:
            engine = preview.create_preview_engine(config)
            for request, level in ((-4, 32), (1, 96), (2, 160), (999, 224)):
                with self.subTest(request=request):
                    actual = preview.render_preview(source, request, config, engine)
                    self.assertEqual(actual.shape, (12, 16, 3))
                    np.testing.assert_allclose(actual.mean(axis=(0, 1)), (level, 20, 40), atol=6)
            load.assert_called_once_with()
        self.assertEqual(set(self.root.rglob("*")), before, "preview wrote output or cache")

    def test_unreadable_real_videos_raise(self):
        missing = self.root / "missing.mp4"
        invalid = self.root / "invalid.mp4"
        invalid.write_bytes(b"not a video")
        for path in (missing, invalid):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "Unable to open video"):
                    preview.probe_source(str(path))


class ImportTests(unittest.TestCase):
    def test_import_and_engine_construction_need_no_gui_cache_or_onnxruntime(self):
        script = """
import builtins
import sys
original_import = builtins.__import__
blocked = {'customtkinter', 'tkinter', 'onnxruntime', 'QualityScaler', 'sqlite3',
           'qualityscaler.gui', 'qualityscaler.widgets', 'qualityscaler.video'}
def guarded_import(name, *args, **kwargs):
    if name in blocked or name.split('.')[0] in blocked:
        raise AssertionError('Unexpected preview dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from qualityscaler.preview import create_preview_engine, PreviewSource
from qualityscaler.config import ProcessingConfig
config = ProcessingConfig([], '', 'Testx2', 1, 0.5, 1, 'Auto', 512,
                          0, False, '.png', '.mp4', 'x264')
engine = create_preview_engine(config)
assert engine.inferenceSession is None
assert engine.input_resize_factor == 0.5
assert not blocked.intersection(sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
