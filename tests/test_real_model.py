"""Opt-in DirectML checks using a locally supplied, ignored ONNX model."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import os
from multiprocessing import get_context
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from qualityscaler.ai import AI_upscale
from qualityscaler.ffmpeg import FFmpeg, UpscaleCancelled
from qualityscaler.frame_store import FrameStore
from qualityscaler.paths import find_by_relative_path
from qualityscaler.video import VideoSettings, VideoUpscaler, decode_frame


@unittest.skipUnless(
    os.environ.get("QUALITYSCALER_TEST_MODEL"),
    "Set QUALITYSCALER_TEST_MODEL to run real-model DirectML integration tests",
)
class RealModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import onnxruntime

        cls.runtime = onnxruntime
        cls.model = os.environ["QUALITYSCALER_TEST_MODEL"]
        model_path = Path(find_by_relative_path(f"AI-onnx/{cls.model}_fp16.onnx"))
        if not model_path.is_file():
            raise RuntimeError(f"Requested integration-test model is missing: {model_path}")
        if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
            raise RuntimeError("Real-model tests require ONNX Runtime's DirectML provider")

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_real_inference_executes_on_directml_and_upscales(self):
        original_session = self.runtime.InferenceSession

        def profiled_session(*args, **kwargs):
            options = kwargs["sess_options"]
            options.enable_profiling = True
            options.profile_file_prefix = str(self.root / "directml-profile")
            return original_session(*args, **kwargs)

        # Instrument only profiling. The production loader still chooses the
        # model, provider, GPU options, and input/output processing.
        with patch("onnxruntime.InferenceSession", side_effect=profiled_session):
            engine = AI_upscale(self.model, "Auto", 1.0, 512)
            engine.ensure_session()
        image = np.random.default_rng(42).integers(
            0, 256, (32, 48, 3), dtype=np.uint8
        )
        try:
            result = engine.AI_orchestration(image)
        finally:
            profile = Path(engine.inferenceSession.end_profiling())
        self.assertEqual(result.shape, (32 * engine.upscale_factor,
                                        48 * engine.upscale_factor, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertGreater(int(result.max()), int(result.min()))
        events = json.loads(profile.read_text(encoding="utf-8"))
        executed_providers = {
            event.get("args", {}).get("provider") for event in events
            if event.get("args", {}).get("provider")
        }
        # get_available_providers()/get_providers() alone cannot prove that
        # inference actually used the GPU instead of a CPU fallback.
        self.assertIn("DmlExecutionProvider", executed_providers)
        print(f"Real model {self.model}: executed providers {sorted(executed_providers)}")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed on PATH")
    def test_real_video_interrupts_resumes_missing_frames_and_preserves_audio(self):
        ffmpeg = FFmpeg(shutil.which("ffmpeg"))
        source = self.root / "real model source.mp4"
        output = self.root / "real model source_upscaled.mp4"
        cache = self.root / "real model source.mp4.qualityscaler" / "frames.sqlite3"
        subprocess.run([
            ffmpeg.executable, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=48x32:rate=4:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
        ], check=True)
        settings = VideoSettings(self.model, workers=2, tiles_resolution=512,
                                 keep_frames=True)
        stop = Event()

        def pause_after_saving(message):
            if message.startswith("Upscaling video"):
                stop.set()

        # Use the production engine factory, actual DirectML sessions, the
        # default checkpoint interval, and real FFmpeg extraction.
        with self.assertRaises(UpscaleCancelled):
            VideoUpscaler(
                ffmpeg=ffmpeg, stop_event=stop, status=pause_after_saving,
            ).run(source, output, cache, settings)
        with FrameStore(cache, read_only=True) as store:
            total = store.get_metadata("total_frames")
            saved_before = store.frame_count("upscaled")
            original_source = list(store.iter_frames("source"))
            saved_frames = {index: store.get_frame("upscaled", index)
                            for index in store.iter_indices("upscaled")}
            self.assertEqual(total, 8)
            self.assertGreater(saved_before, 0)
            self.assertLess(saved_before, total)
            self.assertTrue(store.get_metadata("extraction_complete"))
            self.assertEqual(store.get_metadata("stage"), "paused")
        self.assertFalse(output.exists())

        inference_submissions = []

        class RecordingExecutor(ProcessPoolExecutor):
            def submit(self, *args, **kwargs):
                inference_submissions.append(True)
                return super().submit(*args, **kwargs)

        # Record submissions without replacing inference or process isolation.
        # A complete extraction must not run again or saved JPEGs be recomputed.
        with patch.object(ffmpeg, "extract",
                          side_effect=AssertionError("Resume repeated extraction")):
            VideoUpscaler(
                ffmpeg=ffmpeg,
                executor_factory=partial(RecordingExecutor, mp_context=get_context("spawn")),
            ).run(source, output, cache, settings)
        self.assertEqual(len(inference_submissions), total - saved_before)
        self.assertEqual(list(cache.parent.iterdir()), [cache])
        with FrameStore(cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("stage"), "complete")
            self.assertEqual(store.get_metadata("settings")["model"], self.model)
            self.assertEqual(store.frame_count("source"), total)
            self.assertEqual(store.frame_count("upscaled"), total)
            self.assertEqual(list(store.iter_frames("source")), original_source)
            for index, data in saved_frames.items():
                self.assertEqual(store.get_frame("upscaled", index), data)
            expected_shape = decode_frame(store.get_frame("upscaled", 0)).shape
        factor = 4 if "x4" in self.model else 2 if "x2" in self.model else 1
        self.assertEqual(expected_shape, (32 * factor, 48 * factor, 3))
        capture = cv2.VideoCapture(str(output))
        decoded = 0
        try:
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 4)
            while True:
                success, frame = capture.read()
                if not success:
                    break
                self.assertEqual(frame.shape, expected_shape)
                decoded += 1
        finally:
            capture.release()
        self.assertEqual(decoded, total)
        subprocess.run([
            ffmpeg.executable, "-v", "error", "-i", str(output),
            "-map", "0:a:0", "-f", "null", "-",
        ], check=True)
        inspected = subprocess.run([
            sys.executable, "-m", "qualityscaler", "inspect", str(cache),
        ], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(inspected.stdout)["upscaled_frames"], total)
        print(f"Real model {self.model}: resumed {total - saved_before} missing frames, "
              f"decoded {decoded} frames at {expected_shape[1]}x{expected_shape[0]} with audio")


if __name__ == "__main__":
    unittest.main()
