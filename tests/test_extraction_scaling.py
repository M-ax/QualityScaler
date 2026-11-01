"""CPU-only extraction checks; no models, GPU, or SQLite pipeline required.

QUALITYSCALER_TEST_FFMPEG can point at a bundled FFmpeg in another worktree.
Otherwise use the same bundled/PATH executable discovery as the application.
"""

from io import BytesIO
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from qualityscaler.ffmpeg import FFmpeg, UpscaleCancelled


class ExtractionCommandTests(unittest.TestCase):
    def setUp(self):
        self.ffmpeg = FFmpeg("unused-ffmpeg")
        self.stop = Event()

    def extract_arguments(self, **kwargs):
        frames = [b"first JPEG", b"second JPEG", b"third JPEG"]
        received = []

        def stream_frames(pipe):
            # A callback must consume each frame before the next is requested:
            # extraction must not accumulate a list of full-resolution frames.
            for index, frame in enumerate(frames):
                self.assertEqual(len(received), index)
                yield frame
            self.assertEqual(len(received), len(frames))

        with patch.object(self.ffmpeg, "_pipe") as pipe, patch(
            "qualityscaler.ffmpeg.iter_jpeg_frames", side_effect=stream_frames
        ):
            count = self.ffmpeg.extract(
                Path("source video's ünicode.mkv"), 29.97, self.stop,
                lambda index, data: received.append((index, data)), **kwargs,
            )
        self.assertEqual(count, len(frames))
        self.assertEqual(received, list(enumerate(frames)))
        pipe.assert_called_once()
        self.assertIs(pipe.call_args.args[1], self.stop)
        return pipe.call_args.args[0]

    def test_omitted_and_none_sizes_preserve_exact_legacy_command(self):
        expected = [
            "-threads", "0", "-err_detect", "ignore_err", "-hwaccel", "auto",
            "-i", "source video's ünicode.mkv", "-map", "0:v:0", "-an",
            "-vf", "fps=29.97", "-qscale:v", "3", "-c:v", "mjpeg",
            "-f", "image2pipe", "pipe:1",
        ]
        self.assertEqual(self.extract_arguments(), expected)
        self.assertEqual(self.extract_arguments(input_size=None), expected)

    def test_explicit_dimensions_add_area_scale_after_fps_before_mjpeg(self):
        for size in ((2, 2), (16, 12), (64, 48), (1920, 1080)):
            with self.subTest(size=size):
                arguments = self.extract_arguments(input_size=size)
                filter_index = arguments.index("-vf")
                self.assertEqual(
                    arguments[filter_index + 1],
                    f"fps=29.97,scale={size[0]}:{size[1]}:flags=area",
                )
                self.assertEqual(arguments.count("-vf"), 1)
                self.assertLess(filter_index, arguments.index("-c:v"))
                self.assertEqual(arguments[filter_index + 2:], [
                    "-qscale:v", "3", "-c:v", "mjpeg", "-f", "image2pipe", "pipe:1",
                ])
                self.assertIn("-an", arguments)

    def test_invalid_sizes_are_rejected_before_starting_a_process(self):
        cases = [
            ([], TypeError), ([32, 24], TypeError), ("32:24", TypeError),
            (32, TypeError), (True, TypeError),
            ({"width": 32, "height": 24}, TypeError),
            ((), ValueError), ((32,), ValueError), ((32, 24, 2), ValueError),
        ]
        wrong_types = (
            True, False, None, "32", "32:flags=other", 32.0, 2.5,
            float("nan"), float("inf"), -float("inf"), 2 + 0j, [32],
        )
        for dimension in wrong_types:
            cases += [((dimension, 24), TypeError), ((32, dimension), TypeError)]
        for dimension in (-4, -2, -1, 0, 1, 3, 25):
            cases += [((dimension, 24), ValueError), ((32, dimension), ValueError)]

        save = Mock()
        with patch("qualityscaler.ffmpeg.subprocess.Popen") as popen:
            for size, error in cases:
                with self.subTest(size=size):
                    with self.assertRaisesRegex(error, "input_size"):
                        self.ffmpeg.extract(
                            "unused.mkv", 5, self.stop, save, input_size=size
                        )
            popen.assert_not_called()
        save.assert_not_called()

    def test_size_is_keyword_only(self):
        with patch("qualityscaler.ffmpeg.subprocess.Popen") as popen:
            with self.assertRaises(TypeError):
                self.ffmpeg.extract("unused.mkv", 5, self.stop, Mock(), (16, 12))
            popen.assert_not_called()

    def test_precancelled_scaled_extraction_does_not_start_a_process(self):
        self.stop.set()
        save = Mock()
        with patch("qualityscaler.ffmpeg.subprocess.Popen") as popen:
            with self.assertRaises(UpscaleCancelled):
                self.ffmpeg.extract(
                    "unused.mkv", 5, self.stop, save, input_size=(16, 12)
                )
            popen.assert_not_called()
        save.assert_not_called()

    def test_empty_scaled_stream_preserves_no_frames_error(self):
        with patch.object(self.ffmpeg, "_pipe", return_value=BytesIO()):
            with self.assertRaisesRegex(ValueError, "extracted no video frames"):
                self.ffmpeg.extract(
                    "unused.mkv", 5, self.stop, Mock(), input_size=(16, 12)
                )


FFMPEG_EXECUTABLE = os.environ.get("QUALITYSCALER_TEST_FFMPEG", FFmpeg().executable)


@unittest.skipUnless(Path(FFMPEG_EXECUTABLE).is_file(), "FFmpeg executable is unavailable")
class ExtractionScalingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ffmpeg = FFmpeg(FFMPEG_EXECUTABLE)
        self.levels = [32, 64, 96, 128, 160, 192]

    def make_source(self, size=(32, 24)):
        source = self.root / f"silent video's ünicode {size[0]}x{size[1]}.mkv"
        # Lossless RGB FFV1 preserves odd dimensions and distinct frame values.
        # FFV1 has no hardware decoder, so -hwaccel auto stays on the CPU.
        frames = b"".join(
            np.full((size[1], size[0], 3), level, dtype=np.uint8).tobytes()
            for level in self.levels
        )
        subprocess.run([
            self.ffmpeg.executable, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{size[0]}x{size[1]}", "-framerate", "6",
            "-i", "pipe:0", "-an", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(source),
        ], input=frames, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=15)
        return source

    def decoded_frame(self, data):
        self.assertIsInstance(data, bytes)
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame, "callback did not receive a decodable JPEG")
        return frame

    def extract(self, source, *, fps=6, **kwargs):
        before = set(self.root.iterdir())
        received = []

        def save(index, data):
            frame = self.decoded_frame(data)
            # Inspect actual decoded JPEG pixels inside the extraction callback,
            # not source/container metadata or a requested output size.
            received.append((index, (frame.shape[1], frame.shape[0]), frame.mean()))

        count = self.ffmpeg.extract(source, fps, Event(), save, **kwargs)
        self.assertEqual(count, len(received))
        self.assertEqual([index for index, _, _ in received], list(range(count)))
        self.assertEqual(set(self.root.iterdir()), before, "extraction wrote frame files")
        return received

    def assert_frames(self, received, size, levels):
        self.assertEqual(len(received), len(levels))
        for (_, actual_size, mean), level in zip(received, levels):
            self.assertEqual(actual_size, size)
            self.assertAlmostEqual(mean, level, delta=3)

    def test_scales_silent_even_and_odd_sources_down_and_up_in_order(self):
        for source_size in ((32, 24), (35, 27)):
            source = self.make_source(source_size)
            for input_size in ((16, 12), (64, 48)):
                with self.subTest(source_size=source_size, input_size=input_size):
                    received = self.extract(source, input_size=input_size)
                    self.assert_frames(received, input_size, self.levels)

    def test_omitted_and_none_sizes_keep_even_and_odd_source_dimensions(self):
        for source_size in ((32, 24), (35, 27)):
            source = self.make_source(source_size)
            with self.subTest(source_size=source_size):
                default = self.extract(source)
                explicit_none = self.extract(source, input_size=None)
                self.assert_frames(default, source_size, self.levels)
                self.assertEqual(default, explicit_none)

    def test_scaled_extraction_resamples_fps_without_reordering_frames(self):
        source = self.make_source()
        for fps, expected_count in ((3, 3), (12, 12)):
            with self.subTest(fps=fps):
                # Matroska timestamps and FFmpeg's fps rounding can affect
                # which boundary frame is sampled. Match legacy extraction's
                # temporal choices instead of assuming a particular rounding.
                baseline = self.extract(source, fps=fps)
                self.assertEqual(len(baseline), expected_count)
                levels = [mean for _, _, mean in baseline]
                self.assertEqual(levels, sorted(levels))
                self.assert_frames(
                    self.extract(source, fps=fps, input_size=(16, 12)),
                    (16, 12), levels,
                )

    def test_scaled_cancellation_stops_callbacks_and_reaps_process(self):
        source = self.make_source()
        stop = Event()
        received = []
        processes = []
        original_popen = subprocess.Popen

        def start(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            processes.append(process)
            return process

        def save(index, data):
            frame = self.decoded_frame(data)
            received.append((index, frame.shape))
            stop.set()

        before = set(self.root.iterdir())
        with patch("qualityscaler.ffmpeg.subprocess.Popen", side_effect=start):
            with self.assertRaises(UpscaleCancelled):
                self.ffmpeg.extract(source, 6, stop, save, input_size=(16, 12))
        self.assertEqual(received, [(0, (12, 16, 3))])
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdout.closed)
        self.assertTrue(processes[0].stderr.closed)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_scaled_extraction_preserves_ffmpeg_diagnostics(self):
        with self.assertRaisesRegex(RuntimeError, "FFmpeg failed") as raised:
            self.ffmpeg.extract(
                self.root / "missing.mkv", 6, Event(), Mock(), input_size=(16, 12)
            )
        self.assertIn("missing.mkv", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
