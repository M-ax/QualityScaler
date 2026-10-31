from pathlib import Path
import subprocess
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

    def test_resolution_is_height_then_width(self):
        self.assertEqual(media.get_image_resolution(np.zeros((3, 7, 4))), (3, 7))


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


class VideoAndMetadataTests(unittest.TestCase):
    def test_video_fps_is_positive_and_capture_is_released(self):
        capture = Mock()
        capture.get.return_value = 29.97
        with patch.object(media.cv2, "VideoCapture", return_value=capture) as factory:
            self.assertEqual(media.get_video_fps("input.mp4"), 29.97)
        factory.assert_called_once_with("input.mp4")
        capture.get.assert_called_once_with(cv2.CAP_PROP_FPS)
        capture.release.assert_called_once_with()

    def test_invalid_video_fps_always_releases_capture(self):
        for fps in (0, -1, float("inf"), float("-inf"), float("nan")):
            with self.subTest(fps=fps):
                capture = Mock()
                capture.get.return_value = fps
                with patch.object(media.cv2, "VideoCapture", return_value=capture):
                    with self.assertRaisesRegex(ValueError, "Invalid video frame rate"):
                        media.get_video_fps("broken.mp4")
                capture.release.assert_called_once_with()

    def test_capture_is_released_when_reading_fps_raises(self):
        capture = Mock()
        capture.get.side_effect = RuntimeError("capture failed")
        with patch.object(media.cv2, "VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                media.get_video_fps("broken.mp4")
        capture.release.assert_called_once_with()

    def test_metadata_copy_uses_resource_path_and_argument_list(self):
        with patch.object(media.subprocess, "run") as run:
            media.copy_file_metadata("原本 image.png", "出力 image.png")
        command = run.call_args.args[0]
        expected_exiftool = (
            Path(__file__).resolve().parents[1] / "Assets" / "exiftool.exe"
        )
        self.assertEqual(command[0], str(expected_exiftool))
        self.assertEqual(command[command.index("-TagsFromFile") + 1], "原本 image.png")
        self.assertEqual(command[-1], "出力 image.png")
        self.assertEqual(run.call_args.kwargs, {"check": True, "shell": False})

    def test_metadata_tool_is_best_effort(self):
        for error in (
            FileNotFoundError("exiftool absent"),
            subprocess.CalledProcessError(1, ["exiftool"]),
        ):
            with self.subTest(error=type(error).__name__):
                with patch.object(media.subprocess, "run", side_effect=error):
                    media.copy_file_metadata("original.png", "upscaled.png")


if __name__ == "__main__":
    unittest.main()
