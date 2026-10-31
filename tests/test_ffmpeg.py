from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Timer
from time import monotonic
import unittest
from unittest.mock import patch

import cv2

from qualityscaler.ffmpeg import FFmpeg, UpscaleCancelled
from qualityscaler.frame_store import FrameStore
from qualityscaler.video import VideoSettings, VideoUpscaler, decode_frame


class FFmpegProcessTests(unittest.TestCase):
    def fake_process(self, script):
        original_popen = subprocess.Popen
        processes = []

        def start(command, **kwargs):
            process = original_popen([sys.executable, "-c", script], **kwargs)
            processes.append(process)
            return process

        return patch("qualityscaler.ffmpeg.subprocess.Popen", side_effect=start), processes

    def test_diagnostics_are_drained_and_bounded(self):
        mocked, processes = self.fake_process(
            "import os, sys; os.write(2, b'x' * 2000000); sys.exit(1)"
        )
        with mocked, self.assertRaisesRegex(RuntimeError, "diagnostics omitted") as raised:
            with FFmpeg()._pipe([], Event()) as pipe:
                pipe.read()
        self.assertLess(len(str(raised.exception)), 66000)
        self.assertIsNotNone(processes[0].poll())

    def test_cancel_interrupts_blocked_read_write_and_finalization(self):
        for mode in ("read", "write", "finalize"):
            with self.subTest(mode=mode):
                script = "import time; time.sleep(60)"
                if mode == "finalize":
                    script = "import sys, time; sys.stdin.buffer.read(); time.sleep(60)"
                mocked, processes = self.fake_process(script)
                stop = Event()
                cancellation = Timer(0.25, stop.set)
                cancellation.start()
                started = monotonic()
                try:
                    with mocked, self.assertRaises(UpscaleCancelled):
                        with FFmpeg()._pipe([], stop, writing=mode != "read") as pipe:
                            if mode == "read":
                                pipe.read(1)
                            elif mode == "write":
                                pipe.write(b"x" * (4 * 1024 * 1024))
                finally:
                    cancellation.cancel()
                    cancellation.join()
                self.assertLess(monotonic() - started, 10)
                self.assertIsNotNone(processes[0].poll())


class ResizeEngine:
    """Exercise real video I/O without downloading a model or using a GPU."""

    def calculate_optimal_multithreads_number(self, image, selected):
        return selected

    def AI_orchestration(self, image):
        return cv2.resize(image, (64, 48))


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed on PATH")
class FFmpegIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ffmpeg = FFmpeg(shutil.which("ffmpeg"))
        self.source = self.root / "source video's ünicode.mkv"
        subprocess.run([
            self.ffmpeg.executable, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=32x24:rate=5:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(self.source),
        ], check=True)

    def test_real_extract_cached_upscale_encode_and_resume_with_audio(self):
        cache = self.root / "cache" / "frames.sqlite3"
        output = self.root / "upscaled ünicode.mp4"
        settings = VideoSettings("FakeTestx2", workers=2, keep_frames=True)
        pipeline = VideoUpscaler(ffmpeg=self.ffmpeg,
                                 engine_factory=lambda settings: ResizeEngine(),
                                 executor_factory=ThreadPoolExecutor)
        pipeline.run(self.source, output, cache, settings)
        capture = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 5)
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 5)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 64)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 48)
            self.assertTrue(capture.read()[0])
        finally:
            capture.release()
        # Mapping a required audio stream fails if audio was accidentally lost.
        subprocess.run([
            self.ffmpeg.executable, "-v", "error", "-i", str(output),
            "-map", "0:a:0", "-f", "null", "-",
        ], check=True)
        with FrameStore(cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("stage"), "complete")
            self.assertEqual(store.frame_count("source"), 5)
            self.assertEqual(store.frame_count("upscaled"), 5)
        pipeline.engine_factory = lambda settings: self.fail("Resume loaded an AI model")
        pipeline.run(self.source, output, cache, settings)
        self.assertEqual(list(cache.parent.iterdir()), [cache])
        self.assertFalse(any(self.root.glob(".*.partial-*")))

    def test_failed_encoder_preserves_existing_output_and_removes_partial(self):
        frames = []
        self.ffmpeg.extract(self.source, 5, Event(),
                            lambda index, data: frames.append(data))
        output = self.root / "existing.mp4"
        output.write_bytes(b"previous valid output")
        sentinel = self.root / "existing.partial.mp4"
        sentinel.write_bytes(b"unrelated file")
        with self.assertRaisesRegex(RuntimeError, "FFmpeg failed"):
            self.ffmpeg.encode(
                self.source, output, 5, "not_a_video_codec", [b"not a JPEG"], Event()
            )
        self.assertEqual(output.read_bytes(), b"previous valid output")
        self.assertEqual(sentinel.read_bytes(), b"unrelated file")
        self.assertFalse(any(self.root.glob(".*.partial-*")))

    def test_selected_codec_falls_back_to_x264_with_replayed_cached_frames(self):
        frames = []
        self.ffmpeg.extract(self.source, 5, Event(),
                            lambda index, data: frames.append(data))
        calls = []

        def replay():
            calls.append(True)
            return iter(frames)

        output = self.root / "fallback.mp4"
        messages = []
        self.ffmpeg.encode(
            self.source, output, 5, "not_a_video_codec", replay, Event(),
            output_size=(64, 48), total_frames=5, status=messages.append,
        )
        self.assertEqual(len(calls), 2)
        self.assertTrue(any("x264 fallback" in message for message in messages))
        self.assertTrue(any("100%" in message for message in messages))
        capture = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 5)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 64)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 48)
        finally:
            capture.release()

    def test_precancelled_encoding_never_touches_predictable_partial_file(self):
        output = self.root / "existing.mp4"
        sentinel = self.root / "existing.partial.mp4"
        sentinel.write_bytes(b"unrelated file")
        before = set(self.root.iterdir())
        stop = Event()
        stop.set()
        with self.assertRaises(UpscaleCancelled):
            self.ffmpeg.encode(self.source, output, 5, "x264", [], stop)
        self.assertEqual(sentinel.read_bytes(), b"unrelated file")
        self.assertEqual(set(self.root.iterdir()), before)

    def test_extraction_failure_and_cancellation_propagate(self):
        with self.assertRaisesRegex(RuntimeError, "FFmpeg failed"):
            self.ffmpeg.extract(self.root / "missing.mkv", 5, Event(),
                                lambda index, data: None)
        stop = Event()
        with self.assertRaises(UpscaleCancelled):
            self.ffmpeg.extract(self.source, 5, stop,
                                lambda index, data: stop.set())

    def test_video_without_audio_encodes_successfully(self):
        no_audio = self.root / "silent.mp4"
        subprocess.run([
            self.ffmpeg.executable, "-y", "-v", "error", "-i", str(self.source),
            "-an", "-c:v", "copy", str(no_audio),
        ], check=True)
        frames = []
        self.ffmpeg.extract(no_audio, 5, Event(),
                            lambda index, data: frames.append(data))
        output = self.root / "silent_upscaled.mp4"
        sentinel = self.root / "silent_upscaled.partial.mp4"
        sentinel.write_bytes(b"unrelated file")
        self.ffmpeg.encode(no_audio, output, 5, "x264", frames, Event())
        self.assertTrue(output.is_file())
        self.assertEqual(sentinel.read_bytes(), b"unrelated file")

    def test_cli_resume_recovers_hot_journal_after_process_death(self):
        cache = self.root / "cache" / "frames.sqlite3"
        output = self.root / "recovered.mp4"
        settings = VideoSettings("FakeTestx2", keep_frames=True)
        VideoUpscaler(
            ffmpeg=self.ffmpeg, engine_factory=lambda settings: ResizeEngine(),
            executor_factory=ThreadPoolExecutor,
        ).run(self.source, output, cache, settings)
        # Spill an uncommitted replacement BLOB to disk, then die without
        # closing SQLite. Recovery must restore the prior frame and settings.
        script = """
import os, sys
from qualityscaler.frame_store import FrameStore
store = FrameStore(sys.argv[1])
store.set_metadata("stage", "encoding")
store.commit()
store.set_metadata("settings", {"model": "uncommitted garbage"})
store.put_frame("upscaled", 4, b"not a JPEG" * 1000000)
os._exit(91)
"""
        crashed = subprocess.run([sys.executable, "-c", script, str(cache)])
        self.assertEqual(crashed.returncode, 91)
        self.assertTrue(cache.with_name(cache.name + "-journal").exists())
        resumed = subprocess.run(
            [sys.executable, "-m", "qualityscaler", "resume", str(cache)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        with FrameStore(cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("stage"), "complete")
            self.assertEqual(store.get_metadata("settings")["model"], "FakeTestx2")
            self.assertEqual(store.frame_count("upscaled"), 5)
            self.assertEqual(decode_frame(store.get_frame("upscaled", 4)).shape,
                             (48, 64, 3))
        self.assertFalse(cache.with_name(cache.name + "-journal").exists())


if __name__ == "__main__":
    unittest.main()
