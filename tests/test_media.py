from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from qualityscaler import media


class ImageIOTests(unittest.TestCase):
    def test_lossless_unicode_image_roundtrips(self):
        with TemporaryDirectory() as directory:
            for channels, dtype in ((None, np.uint8), (3, np.uint8), (4, np.uint8),
                                    (None, np.uint16)):
                with self.subTest(channels=channels, dtype=dtype):
                    shape = (5, 7) if channels is None else (5, 7, channels)
                    image = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
                    target = Path(directory) / f"画像 🐈 {channels} {np.dtype(dtype)}.png"
                    media.image_write(str(target), image, ".png")
                    actual = media.image_read(str(target))
                    np.testing.assert_array_equal(actual, image)
                    self.assertEqual(actual.dtype, image.dtype)

    def test_selected_extension_controls_encoding(self):
        image = np.full((6, 8, 3), 128, dtype=np.uint8)
        with TemporaryDirectory() as directory:
            target = Path(directory) / "selected.jpg"
            media.image_write(str(target), image, ".png")
            self.assertTrue(target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            np.testing.assert_array_equal(media.image_read(str(target)), image)

    def test_default_encoding_is_jpeg(self):
        image = np.full((6, 8, 3), 128, dtype=np.uint8)
        with TemporaryDirectory() as directory:
            target = Path(directory) / "default.jpg"
            media.image_write(str(target), image)
            self.assertTrue(target.read_bytes().startswith(b"\xff\xd8"))
            np.testing.assert_allclose(media.image_read(str(target)), image, atol=1)

    def test_default_codec_options_are_selected_case_insensitively(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for extension, options in (
            (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
            (".JPEG", [cv2.IMWRITE_JPEG_QUALITY, 95]),
            (".png", [cv2.IMWRITE_PNG_COMPRESSION, 1]),
            (".PNG", [cv2.IMWRITE_PNG_COMPRESSION, 1]),
            (".bmp", []),
        ):
            with self.subTest(extension=extension), TemporaryDirectory() as directory:
                target = str(Path(directory) / "encoded image")
                with patch.object(media.cv2, "imencode", wraps=cv2.imencode) as encode:
                    media.image_write(target, image, extension)
                encode.assert_called_once_with(extension, image, options)

    def test_codec_options_can_be_configured(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for extension, options in (
            (".jpeg", [cv2.IMWRITE_JPEG_QUALITY, 83]),
            (".PNG", [cv2.IMWRITE_PNG_COMPRESSION, 7]),
            (".bmp", []),
        ):
            with self.subTest(extension=extension), TemporaryDirectory() as directory:
                target = str(Path(directory) / "encoded image")
                with patch.object(media.cv2, "imencode", wraps=cv2.imencode) as encode:
                    media.image_write(
                        target, image, extension, jpeg_quality=83, png_compression=7
                    )
                encode.assert_called_once_with(extension, image, options)

    def test_invalid_and_empty_images_raise_decode_errors(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "invalid.png"
            for data in (b"", b"not an image"):
                with self.subTest(data=data):
                    target.write_bytes(data)
                    with self.assertRaisesRegex(ValueError, "Unable to decode"):
                        media.image_read(str(target))

    def test_missing_input_still_raises_file_error(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                media.image_read(str(Path(directory) / "missing.png"))

    def test_real_encode_failure_does_not_overwrite_existing_file(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "existing.png"
            target.write_bytes(b"keep original")
            for image, extension in (
                (np.zeros((0, 0, 3), dtype=np.uint8), ".png"),
                (np.zeros((2, 2, 3), dtype=np.uint8), ".unsupported"),
            ):
                with self.subTest(extension=extension):
                    with self.assertRaisesRegex(ValueError, "Unable to encode"):
                        media.image_write(str(target), image, extension)
                    self.assertEqual(target.read_bytes(), b"keep original")

    def test_false_or_empty_encode_result_is_rejected_before_writing(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "missing.png"
            for result in (
                (False, np.array([1, 2], dtype=np.uint8)),
                (True, None),
                (True, np.array([], dtype=np.uint8)),
            ):
                with self.subTest(result=result):
                    with patch.object(media.cv2, "imencode", return_value=result):
                        with self.assertRaisesRegex(ValueError, "Unable to encode"):
                            media.image_write(str(target), np.zeros((2, 2, 3)), ".png")
                    self.assertFalse(target.exists())

    def test_decode_exception_has_path_context(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "image.png"
            target.write_bytes(b"corrupt")
            with patch.object(media.cv2, "imdecode", side_effect=cv2.error("bad")):
                with self.assertRaisesRegex(ValueError, "image.png"):
                    media.image_read(str(target))


class ImageGeometryTests(unittest.TestCase):
    def test_resolution_is_height_then_width(self):
        self.assertEqual(media.get_image_resolution(np.zeros((3, 7, 4))), (3, 7))

    def test_image_modes_do_not_convert_or_mutate_inputs(self):
        for shape, mode in (
            ((3, 7), "Grayscale"),
            ((3, 7, 3), "RGB"),
            ((3, 7, 4), "RGBA"),
            ((3, 7, 1), None),
            ((3, 7, 2), None),
            ((3,), None),
        ):
            with self.subTest(shape=shape):
                image = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
                original = image.copy()
                self.assertEqual(media.get_image_mode(image), mode)
                np.testing.assert_array_equal(image, original)

    def test_resize_truncates_then_rounds_up_to_even_dimensions(self):
        image = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
        original = image.copy()
        for factor, dimensions in (
            (0.5, (6, 4)),
            (1, (12, 8)),
            (1.3, (14, 10)),
            (2, (22, 14)),
        ):
            for interpolation in (cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_CUBIC):
                with self.subTest(factor=factor, interpolation=interpolation):
                    result = media.resize_with_factor(image, factor, interpolation)
                    self.assertEqual(result.shape[:2], dimensions[::-1])
                    np.testing.assert_array_equal(
                        result,
                        cv2.resize(image, dimensions, interpolation=interpolation),
                    )
        np.testing.assert_array_equal(image, original)

    def test_scaled_dimensions_is_pure_and_rounds_even_at_unit_scale(self):
        image = np.zeros((7, 11, 3), dtype=np.uint8)
        with patch.object(media.cv2, "resize", side_effect=AssertionError("not pure")):
            for factor, dimensions in (
                (0.5, (6, 4)),
                (1, (12, 8)),
                (1.3, (14, 10)),
                (2, (22, 14)),
            ):
                with self.subTest(factor=factor):
                    self.assertEqual(media.scaled_dimensions(image, factor), dimensions)

    def test_unit_resize_still_invokes_opencv_with_even_dimensions(self):
        image = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
        with patch.object(media.cv2, "resize", wraps=cv2.resize) as resize:
            result = media.resize_with_factor(image, 1, cv2.INTER_CUBIC)
        resize.assert_called_once_with(image, (12, 8), interpolation=cv2.INTER_CUBIC)
        self.assertEqual(result.shape, (8, 12, 3))
        self.assertEqual(media.resize_with_output_factor(image, 1).shape, (8, 12, 3))

    def test_output_resize_always_uses_linear_interpolation(self):
        image = np.arange(36, dtype=np.uint8).reshape(6, 6)
        for factor, dimensions in ((2 / 3, (4, 4)), (1, (6, 6)), (2, (12, 12))):
            with self.subTest(factor=factor):
                result = media.resize_with_output_factor(image, factor)
                np.testing.assert_array_equal(
                    result,
                    cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR),
                )
        expected = np.zeros((4, 4), dtype=np.uint8)
        with patch.object(media, "resize_with_factor", return_value=expected) as resize:
            self.assertIs(media.resize_with_output_factor(image, 0.5), expected)
        resize.assert_called_once_with(image, 0.5, cv2.INTER_LINEAR)

    def test_resize_preserves_uint16_and_alpha_channels(self):
        image = np.arange(3 * 5 * 4, dtype=np.uint16).reshape(3, 5, 4)
        result = media.resize_with_output_factor(image, 2)
        self.assertEqual(result.shape, (6, 10, 4))
        self.assertEqual(result.dtype, np.uint16)


class SharpeningTests(unittest.TestCase):
    def test_sharpen_matches_upstream_unsharp_mask_without_mutating_input(self):
        for shape in ((7, 9), (7, 9, 3), (7, 9, 4)):
            for dtype in (np.uint8, np.uint16):
                image = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
                original = image.copy()
                for amount in (0, 0.3, 0.5):
                    with self.subTest(shape=shape, dtype=dtype, amount=amount):
                        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
                        expected = cv2.addWeighted(
                            image, 1 + amount, blurred, -amount, 0, dtype=cv2.CV_8U
                        )
                        result = media.sharpen_image(image, amount)
                        np.testing.assert_array_equal(result, expected)
                        self.assertEqual(result.dtype, np.uint8)
                        np.testing.assert_array_equal(image, original)

    def test_opencv_errors_fall_back_to_unsharpened_output(self):
        image = np.full((4, 6, 3), 128, dtype=np.uint8)
        for operation in ("GaussianBlur", "addWeighted"):
            with self.subTest(operation=operation):
                with patch.object(media.cv2, operation, side_effect=cv2.error("bad")):
                    self.assertIs(media.sharpen_image(image, 0.5), image)

    def test_save_wrapper_uses_pure_sharpen_result(self):
        image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        expected = media.sharpen_image(image, 0.3)
        with TemporaryDirectory() as directory:
            target = str(Path(directory) / "鮮明.png")
            with patch.object(media, "sharpen_image", return_value=expected) as sharpen:
                media.sharpen_and_save(target, image, 0.3, ".png")
            sharpen.assert_called_once_with(image, 0.3)
            np.testing.assert_array_equal(media.image_read(target), expected)

    def test_save_wrapper_defaults_to_jpeg(self):
        image = np.full((4, 6, 3), 128, dtype=np.uint8)
        with patch.object(media, "image_write") as write:
            media.sharpen_and_save("output.jpg", image, 0.3)
        self.assertEqual(write.call_args.args[0], "output.jpg")
        self.assertEqual(write.call_args.args[2], ".jpg")
        np.testing.assert_array_equal(write.call_args.args[1], image)

    def test_save_wrapper_writes_original_after_opencv_error(self):
        image = np.full((4, 6, 3), 128, dtype=np.uint8)
        with patch.object(media.cv2, "GaussianBlur", side_effect=cv2.error("bad")):
            with patch.object(media, "image_write") as write:
                media.sharpen_and_save("output.png", image, 0.5, ".png")
        write.assert_called_once_with("output.png", image, ".png")

    def test_save_errors_propagate_without_retry(self):
        image = np.full((4, 6, 3), 128, dtype=np.uint8)
        for error in (OSError("disk full"), ValueError("Unable to encode")):
            with self.subTest(error=type(error).__name__):
                with patch.object(media, "image_write", side_effect=error) as write:
                    with self.assertRaises(type(error)):
                        media.sharpen_and_save("output.png", image, 0.5, ".png")
                write.assert_called_once()


class BlendingTests(unittest.TestCase):
    def test_blending_resizes_without_mutating_inputs(self):
        for starting_shape, target_shape in (((3, 5, 3), (6, 10, 3)),
                                             ((6, 10, 3), (3, 4, 3))):
            with self.subTest(starting_shape=starting_shape):
                starting = np.arange(np.prod(starting_shape), dtype=np.uint8).reshape(
                    starting_shape
                )
                upscaled = np.full(target_shape, 160, dtype=np.uint8)
                original_starting = starting.copy()
                original_upscaled = upscaled.copy()
                expected = cv2.addWeighted(
                    cv2.resize(starting, (target_shape[1], target_shape[0])),
                    0.3, upscaled, 0.7, 0,
                )
                np.testing.assert_array_equal(
                    media.blend_images(starting, upscaled, 0.3), expected
                )
                np.testing.assert_array_equal(starting, original_starting)
                np.testing.assert_array_equal(upscaled, original_upscaled)

    def test_legacy_downsize_interpolation_is_linear_not_area(self):
        starting = np.arange(36, dtype=np.uint8).reshape(6, 6)
        upscaled = np.zeros((4, 4), dtype=np.uint8)
        result = media.blend_images(starting, upscaled, 1)
        np.testing.assert_array_equal(result, cv2.resize(starting, (4, 4)))
        self.assertFalse(
            np.array_equal(
                result, cv2.resize(starting, (4, 4), interpolation=cv2.INTER_AREA)
            )
        )

    def test_grayscale_and_blending_extremes(self):
        starting = np.full((2, 3), 50, dtype=np.uint8)
        upscaled = np.full((4, 6), 150, dtype=np.uint8)
        for factor, expected in ((0, 150), (0.5, 100), (1, 50)):
            with self.subTest(factor=factor):
                np.testing.assert_array_equal(
                    media.blend_images(starting, upscaled, factor),
                    np.full((4, 6), expected, dtype=np.uint8),
                )

    def test_rgba_starting_image_adds_opaque_alpha_to_rgb_upscaled(self):
        starting = np.full((2, 3, 4), (20, 40, 60, 100), dtype=np.uint8)
        upscaled = np.full((4, 6, 3), (60, 80, 100), dtype=np.uint8)
        expected_upscaled = np.concatenate(
            (upscaled, np.full((4, 6, 1), 255, dtype=np.uint8)), axis=2
        )
        expected = cv2.addWeighted(
            cv2.resize(starting, (6, 4)), 0.5, expected_upscaled, 0.5, 0
        )
        np.testing.assert_array_equal(
            media.blend_images(starting, upscaled, 0.5), expected
        )
        self.assertEqual(upscaled.shape, (4, 6, 3))

    def test_rgba_blends_alpha_as_well_as_color(self):
        starting = np.full((2, 3, 4), (20, 40, 60, 100), dtype=np.uint8)
        upscaled = np.full((4, 6, 4), (60, 80, 100, 200), dtype=np.uint8)
        expected = np.full((4, 6, 4), (40, 60, 80, 150), dtype=np.uint8)
        np.testing.assert_array_equal(
            media.blend_images(starting, upscaled, 0.5), expected
        )

    def test_incompatible_modes_fall_back_to_upscaled_image(self):
        for starting_shape, upscaled_shape in (
            ((2, 3), (4, 6, 3)),
            ((2, 3, 3), (4, 6, 4)),
            ((2, 3, 4), (4, 6)),
        ):
            with self.subTest(starting=starting_shape, upscaled=upscaled_shape):
                starting = np.zeros(starting_shape, dtype=np.uint8)
                upscaled = np.full(upscaled_shape, 128, dtype=np.uint8)
                self.assertIs(media.blend_images(starting, upscaled, 0.5), upscaled)

    def test_save_wrapper_uses_pure_blend_result(self):
        starting = np.full((2, 3, 3), 50, dtype=np.uint8)
        upscaled = np.full((4, 6, 3), 150, dtype=np.uint8)
        with TemporaryDirectory() as directory:
            target = str(Path(directory) / "混合.png")
            expected = media.blend_images(starting, upscaled, 0.5)
            with patch.object(media, "blend_images", return_value=expected) as blend:
                media.blend_images_and_save(target, starting, upscaled, 0.5, ".png")
            blend.assert_called_once_with(starting, upscaled, 0.5)
            np.testing.assert_array_equal(media.image_read(target), expected)

    def test_save_failure_is_not_hidden_by_blending_fallback(self):
        image = np.full((2, 3, 3), 128, dtype=np.uint8)
        with patch.object(media, "image_write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                media.blend_images_and_save("output.png", image, image, 0.5, ".png")


class VideoFPSTests(unittest.TestCase):
    def test_sanitize_preserves_finite_positive_frame_rates(self):
        for fps in (0.001, 1, 23.976, 29.97, 60, 240):
            with self.subTest(fps=fps):
                self.assertEqual(media.sanitize_fps(fps), fps)

    def test_sanitize_falls_back_for_invalid_frame_rates(self):
        for fps in (0, -1, float("inf"), float("-inf"), float("nan"), None, "unknown"):
            with self.subTest(fps=fps):
                self.assertEqual(media.sanitize_fps(fps), 30.0)

    def test_video_fps_is_positive_and_capture_is_released(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 29.97
        with patch.object(media.cv2, "VideoCapture", return_value=capture) as factory:
            self.assertEqual(media.get_video_fps("input.mp4"), 29.97)
        factory.assert_called_once_with("input.mp4")
        capture.isOpened.assert_called_once_with()
        capture.get.assert_called_once_with(cv2.CAP_PROP_FPS)
        capture.release.assert_called_once_with()

    def test_unknown_fps_of_open_video_falls_back_and_releases_capture(self):
        for fps in (0, -1, float("inf"), float("-inf"), float("nan")):
            with self.subTest(fps=fps):
                capture = Mock()
                capture.isOpened.return_value = True
                capture.get.return_value = fps
                with patch.object(media.cv2, "VideoCapture", return_value=capture):
                    self.assertEqual(media.get_video_fps("valid.mp4"), 30.0)
                capture.isOpened.assert_called_once_with()
                capture.release.assert_called_once_with()

    def test_unopened_video_is_rejected_even_with_plausible_fps(self):
        capture = Mock()
        capture.isOpened.return_value = False
        capture.get.return_value = 30
        with patch.object(media.cv2, "VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "Unable to open video: broken.mp4"):
                media.get_video_fps("broken.mp4")
        capture.get.assert_not_called()
        capture.release.assert_called_once_with()

    def test_missing_and_unreadable_videos_are_rejected(self):
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.mp4"
            invalid = Path(directory) / "invalid.mp4"
            invalid.write_bytes(b"not a video")
            for path in (missing, invalid):
                with self.subTest(path=path):
                    with self.assertRaisesRegex(ValueError, "Unable to open video"):
                        media.get_video_fps(str(path))

    def test_capture_is_released_when_reading_fps_raises(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = RuntimeError("capture failed")
        with patch.object(media.cv2, "VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                media.get_video_fps("broken.mp4")
        capture.release.assert_called_once_with()

    def test_capture_is_released_when_checking_open_state_raises(self):
        capture = Mock()
        capture.isOpened.side_effect = RuntimeError("capture failed")
        with patch.object(media.cv2, "VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                media.get_video_fps("broken.mp4")
        capture.release.assert_called_once_with()


class ImportTests(unittest.TestCase):
    def test_helpers_do_not_require_gui_inference_or_windows_integrations(self):
        script = """
import builtins
import sys
original_import = builtins.__import__
blocked = {'customtkinter', 'tkinter', 'onnxruntime', 'QualityScaler',
           'winotify', 'win32api', 'win32con', 'pythoncom'}
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('Unexpected helper dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import qualityscaler.media
import qualityscaler.paths
assert not blocked.intersection(sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
