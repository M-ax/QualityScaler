from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import sleep
import sys
import unittest
from unittest.mock import patch
from weakref import WeakSet

import cv2
import numpy as np

from qualityscaler.ffmpeg import UpscaleCancelled
from qualityscaler.frame_store import FrameStore
from qualityscaler.job_lock import JobAlreadyRunning, JobLock
from qualityscaler.video import (
    VideoSettings, VideoUpscaler, decode_frame, get_resume_progress,
    initialize_worker, process_frame,
)


def jpeg(value):
    return cv2.imencode(".jpg", np.full((8, 12, 3), value, np.uint8))[1].tobytes()


class FakeFFmpeg:
    def __init__(self):
        self.extractions = 0
        self.encodings = 0
        self.frames = [jpeg(value) for value in (20, 40, 60, 80, 100)]
        self.encoded_values = []
        self.fail_encoding = False
        self.stop_during_extraction = False

    def extract(self, source, fps, stop, save_frame):
        self.extractions += 1
        for index, data in enumerate(self.frames):
            save_frame(index, data)
            if self.stop_during_extraction and index == 1:
                stop.set()
        return len(self.frames)

    def encode(self, source, output, fps, codec, frames, stop, **options):
        self.encodings += 1
        self.encoding_options = options
        if self.fail_encoding:
            raise RuntimeError("Test encoder failed")
        if callable(frames):
            frames = frames()
        self.encoded_values = [round(float(decode_frame(data).mean())) for data in frames]
        Path(output).write_bytes(b"encoded video")


class FakeEngine:
    def __init__(self, calls, fail_value=None):
        self.calls = calls
        self.fail_value = fail_value

    def calculate_optimal_multithreads_number(self, image, selected):
        assert image.shape == (8, 12, 3)
        return selected

    def AI_orchestration(self, image):
        value = round(float(image.mean()))
        self.calls.append(value)
        if value == self.fail_value:
            raise RuntimeError("Test inference failed")
        return cv2.resize(image, (24, 16))


class CrashingEngine:
    """A child process dying must not take the SQLite writer down with it."""

    def __init__(self, settings):
        pass

    def calculate_optimal_multithreads_number(self, image, selected):
        return selected

    def AI_orchestration(self, image):
        os._exit(71)


class SpawnEngine:
    """Log process ownership and per-session calls without requiring a GPU."""

    def __init__(self, settings):
        self.log = Path(os.environ["QUALITYSCALER_TEST_WORKER_LOG"]) / f"{os.getpid()}.jsonl"
        self.calls = 0
        self.record("init", gui_loaded=any(
            name in sys.modules for name in ("qualityscaler.gui", "customtkinter", "tkinter")
        ))

    def record(self, event, **values):
        with self.log.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"event": event, "calls": self.calls, **values}) + "\n")

    def AI_orchestration(self, image):
        self.calls += 1
        self.record("infer")
        return cv2.resize(image, (24, 16))


class VideoUpscalerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "a video's input.mp4"
        self.source.write_bytes(b"source video")
        self.output = self.directory / "input_upscaled.mp4"
        self.cache = self.directory / "a video's input.mp4.qualityscaler" / "frames.sqlite3"
        self.ffmpeg = FakeFFmpeg()
        self.stop = Event()
        self.calls = []
        self.settings = VideoSettings("Testx2", workers=2, keep_frames=True)
        patcher = patch("qualityscaler.video.get_video_fps", return_value=24.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def upscaler(self, **kwargs):
        # Deterministic in-process fake engines keep unit tests GPU-free.
        # The production executor uses spawned processes instead.
        kwargs.setdefault("executor_factory", ThreadPoolExecutor)
        return VideoUpscaler(
            ffmpeg=self.ffmpeg, stop_event=self.stop,
            engine_factory=lambda settings: FakeEngine(self.calls),
            checkpoint_interval=1, **kwargs,
        )

    def run_job(self, upscaler=None, settings=None):
        return (upscaler or self.upscaler()).run(
            self.source, self.output, self.cache, settings or self.settings
        )

    def test_single_cache_stores_settings_frames_and_ordered_output(self):
        self.run_job()
        self.assertEqual(self.ffmpeg.encoded_values, [20, 40, 60, 80, 100])
        self.assertCountEqual(self.calls, self.ffmpeg.encoded_values)
        self.assertEqual(list(self.cache.parent.iterdir()), [self.cache])
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.frame_count("source"), 5)
            self.assertEqual(store.frame_count("upscaled"), 5)
            self.assertEqual(store.get_metadata("stage"), "complete")
            self.assertEqual(store.get_metadata("settings")["model"], "Testx2")
            self.assertEqual(store.get_metadata("settings")["codec"], "x264")
            self.assertEqual(store.get_metadata("total_frames"), 5)

    def test_video_geometry_preserves_upstream_aspect_derived_rounding(self):
        image = np.full((11, 15, 3), 20, np.uint8)
        self.ffmpeg.frames = [cv2.imencode(".jpg", image)[1].tobytes()]
        self.run_job(settings=replace(
            self.settings, input_resize_factor=0.5, output_resize_factor=1.25,
        ))
        # Upstream task: input8x6 -> output20x16; independently scaling the
        # source height would incorrectly produce14 rather than16.
        self.assertEqual(self.ffmpeg.encoding_options["output_size"], (20, 16))

    def test_video_sharpens_native_frames_before_jpeg_and_defers_output_resize(self):
        data = jpeg(20)
        original = decode_frame(data)
        native = np.full((16, 24, 3), 40, np.uint8)
        blended = np.full_like(native, 50)
        sharpened = np.full_like(native, 60)
        events = []
        engine = FakeEngine([])

        def infer(image):
            self.assertEqual(image.shape, original.shape)
            events.append("AI")
            return native

        def blend(starting, image, amount):
            self.assertIs(image, native)
            self.assertEqual(amount, 0.3)
            events.append("blend")
            return blended

        def sharpen(image, amount):
            self.assertIs(image, blended)
            self.assertEqual(amount, 0.5)
            events.append("sharpen")
            return sharpened

        def encode(extension, image, params):
            self.assertIs(image, sharpened)
            self.assertEqual(params, [cv2.IMWRITE_JPEG_QUALITY, 95])
            events.append("JPEG")
            return True, np.array([1, 2, 3], np.uint8)

        engine.AI_orchestration = infer
        settings = replace(self.settings, blending_factor=0.3, sharpening_amount=0.5,
                           output_resize_factor=0.25)
        initialize_worker(settings, lambda settings: engine)
        with patch("qualityscaler.video.blend_images", side_effect=blend), \
                patch("qualityscaler.video.sharpen_image", side_effect=sharpen), \
                patch("qualityscaler.video.cv2.imencode", side_effect=encode):
            self.assertEqual(process_frame(data), b"\x01\x02\x03")
        self.assertEqual(events, ["AI", "blend", "sharpen", "JPEG"])
        initialize_worker(self.settings, lambda settings: engine)
        with patch("qualityscaler.video.cv2.imencode",
                   return_value=(True, np.array([1], np.uint8))) as encoded:
            process_frame(data)
        self.assertEqual(encoded.call_args.args[2], [cv2.IMWRITE_JPEG_QUALITY, 90])

    def test_interrupted_upscaling_resumes_only_missing_frames(self):
        def status(message):
            if message.startswith("Upscaling video"):
                self.stop.set()

        with self.assertRaises(UpscaleCancelled):
            self.run_job(self.upscaler(status=status))
        with FrameStore(self.cache, read_only=True) as store:
            saved = list(store.iter_indices("upscaled"))
            self.assertGreater(len(saved), 0)
            self.assertLess(len(saved), 5)
            self.assertEqual(store.get_metadata("stage"), "paused")
        self.assertFalse(self.output.exists())
        self.stop.clear()
        self.calls.clear()
        self.run_job()
        self.assertEqual(self.ffmpeg.extractions, 1)
        self.assertCountEqual(self.calls, [
            value for index, value in enumerate((20, 40, 60, 80, 100))
            if index not in saved
        ])
        self.assertEqual(self.ffmpeg.encoded_values, [20, 40, 60, 80, 100])

    def test_partial_extraction_replays_and_reuses_committed_frames(self):
        self.ffmpeg.stop_during_extraction = True
        with self.assertRaises(UpscaleCancelled):
            self.run_job()
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.frame_count("source"), 2)
            first_frame = store.get_frame("source", 0)
            self.assertFalse(store.get_metadata("extraction_complete", False))
        self.ffmpeg.stop_during_extraction = False
        self.stop.clear()
        self.run_job()
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.get_frame("source", 0), first_frame)
            self.assertTrue(store.get_metadata("extraction_complete"))
        self.assertEqual(self.ffmpeg.extractions, 2)

    def test_out_of_order_workers_still_encode_in_frame_order(self):
        finished = []

        class DelayedEngine(FakeEngine):
            def AI_orchestration(self, image):
                value = round(float(image.mean()))
                if value == 20:
                    sleep(0.1)
                result = super().AI_orchestration(image)
                finished.append(value)
                return result

        pipeline = VideoUpscaler(
            ffmpeg=self.ffmpeg, engine_factory=lambda settings: DelayedEngine(self.calls),
            executor_factory=ThreadPoolExecutor,
        )
        self.run_job(pipeline)
        self.assertNotEqual(finished, [20, 40, 60, 80, 100])
        self.assertEqual(self.ffmpeg.encoded_values, [20, 40, 60, 80, 100])

    def test_long_video_submissions_are_bounded(self):
        self.ffmpeg.frames = [jpeg(20)] * 1000
        submitted = []

        class RecordingExecutor(ThreadPoolExecutor):
            def submit(self, *args, **kwargs):
                submitted.append(True)
                return super().submit(*args, **kwargs)

        def status(message):
            if message.startswith("Upscaling video"):
                self.stop.set()

        with self.assertRaises(UpscaleCancelled):
            self.run_job(self.upscaler(status=status, executor_factory=RecordingExecutor))
        self.assertLessEqual(len(submitted), 2 * self.settings.workers)

    def test_million_frame_source_keeps_refill_cycles_bounded(self):
        class LazyStore:
            def __init__(self):
                self.metadata = {"total_frames": 1000000}
                self.saved = set()
                self.data = jpeg(20)

            def get_metadata(self, key):
                return self.metadata.get(key)

            def set_metadata(self, key, value):
                self.metadata[key] = value

            def frame_count(self, kind):
                return 1000000 if kind == "source" else len(self.saved)

            def get_frame(self, kind, index):
                return self.data

            def has_frame(self, kind, index):
                return index in self.saved

            def put_frame(self, kind, index, data):
                self.saved.add(index)

            def commit(self):
                pass

        futures = WeakSet()
        peaks = {"retained": 0, "unfinished": 0, "submitted": 0}

        class RecordingExecutor(ThreadPoolExecutor):
            def submit(self, *args, **kwargs):
                future = super().submit(*args, **kwargs)
                futures.add(future)
                peaks["submitted"] += 1
                peaks["retained"] = max(peaks["retained"], len(futures))
                peaks["unfinished"] = max(
                    peaks["unfinished"], sum(not item.done() for item in futures)
                )
                return future

        store = LazyStore()

        def status(message):
            if len(store.saved) >= 128:
                self.stop.set()

        with self.assertRaises(UpscaleCancelled):
            self.upscaler(status=status, executor_factory=RecordingExecutor)._upscale(
                store, self.settings
            )
        self.assertGreaterEqual(peaks["submitted"], 128)
        self.assertLessEqual(peaks["unfinished"], self.settings.workers * 2)
        # Current pending plus the previous wait() result can coexist briefly.
        self.assertLessEqual(peaks["retained"], self.settings.workers * 4 + 2)

    def test_encoding_failure_retains_frames_for_codec_retry(self):
        self.ffmpeg.fail_encoding = True
        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            self.run_job()
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("stage"), "failed")
            self.assertEqual(store.get_metadata("interrupted_stage"), "encoding")
            self.assertEqual(store.frame_count("upscaled"), 5)
        self.ffmpeg.fail_encoding = False
        self.calls.clear()
        self.run_job(settings=replace(self.settings, codec="x265", workers=1))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ffmpeg.extractions, 1)
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("settings")["codec"], "x265")

    def test_default_batch_is_committed_before_waiting_for_running_inference(self):
        blocked = Event()
        release = Event()
        stopping = Event()
        errors = []

        class BlockingEngine(FakeEngine):
            def AI_orchestration(self, image):
                value = round(float(image.mean()))
                if value == 20:
                    blocked.wait(5)
                if value == 40:
                    blocked.set()
                    release.wait(10)
                return super().AI_orchestration(image)

        def status(message):
            if message.startswith("Upscaling video"):
                self.stop.set()
                stopping.set()

        pipeline = VideoUpscaler(
            ffmpeg=self.ffmpeg, stop_event=self.stop, status=status,
            engine_factory=lambda settings: BlockingEngine(self.calls),
            executor_factory=ThreadPoolExecutor,
        )

        def run():
            try:
                self.run_job(pipeline)
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=run)
        thread.start()
        try:
            self.assertTrue(stopping.wait(5))
            # Wait for the cancellation path's explicit commit, without
            # releasing the blocked model call first.
            for _ in range(100):
                with FrameStore(self.cache, read_only=True) as store:
                    if store.frame_count("upscaled"):
                        break
                sleep(0.01)
            else:
                self.fail("Saved results were not committed while inference blocked")
            self.assertTrue(thread.is_alive())
        finally:
            release.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], UpscaleCancelled)
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("stage"), "paused")

    def test_changed_processing_settings_leave_existing_job_untouched(self):
        self.run_job()
        with self.assertRaisesRegex(ValueError, "different AI"):
            self.run_job(settings=replace(self.settings, blending_factor=0.5))
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("stage"), "complete")
            self.assertEqual(store.get_metadata("settings")["blending_factor"], 0)
            self.assertEqual(store.frame_count("upscaled"), 5)

    def test_sharpening_changes_invalidate_saved_processing_settings(self):
        self.run_job()
        with self.assertRaisesRegex(ValueError, "sharpening"):
            self.run_job(settings=replace(self.settings, sharpening_amount=0.5))

    def test_changed_model_weights_cannot_be_mixed_into_a_resumed_job(self):
        with patch("qualityscaler.video.model_digest", return_value="original-weights"):
            self.run_job()
        with patch("qualityscaler.video.model_digest", return_value="updated-weights"):
            with self.assertRaisesRegex(ValueError, "different AI"):
                self.run_job()

    def test_legacy_processing_cache_is_not_silently_reused_by_new_engine(self):
        self.run_job()
        with FrameStore(self.cache) as store:
            store.set_metadata("job_version", 1)
        with self.assertRaisesRegex(ValueError, "different processing version"):
            self.run_job()

    def test_resume_badge_uses_sqlite_source_and_processing_identity(self):
        # The GUI and backend use the same stable directory.
        self.cache = self.directory / (self.source.name + ".qualityscaler") / "frames.sqlite3"
        self.run_job()
        self.assertEqual(get_resume_progress(self.source, str(self.directory), self.settings), 100)
        self.assertIsNone(get_resume_progress(
            self.source, str(self.directory), replace(self.settings, sharpening_amount=0.5)
        ))
        self.source.write_bytes(b"modified source")
        self.assertIsNone(get_resume_progress(self.source, str(self.directory), self.settings))

    def test_modified_source_is_rejected_without_resetting_cache(self):
        self.run_job()
        self.source.write_bytes(b"a modified source")
        with self.assertRaisesRegex(ValueError, "modified source"):
            self.run_job()
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.frame_count("source"), 5)
            self.assertEqual(store.get_metadata("stage"), "complete")

    def test_competing_runner_is_rejected_before_modifying_metadata(self):
        self.run_job()
        before = self.cache.read_bytes()
        with JobLock(self.cache):
            with self.assertRaisesRegex(JobAlreadyRunning, "already running"):
                self.run_job(settings=replace(self.settings, codec="x265"))
        self.assertEqual(self.cache.read_bytes(), before)
        self.assertEqual(self.ffmpeg.encodings, 1)

    def test_inference_failure_is_not_reported_as_success(self):
        upscaler = VideoUpscaler(
            ffmpeg=self.ffmpeg,
            engine_factory=lambda settings: FakeEngine(self.calls, fail_value=60),
            checkpoint_interval=1,
            executor_factory=ThreadPoolExecutor,
        )
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_job(upscaler)
        self.assertEqual(self.ffmpeg.encodings, 0)
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("stage"), "failed")
            self.assertFalse(store.has_frame("upscaled", 2))
        self.calls.clear()
        self.run_job()
        self.assertIn(60, self.calls)

    def test_native_worker_death_keeps_writer_and_cache_available_for_resume(self):
        with self.assertRaises(BrokenProcessPool):
            self.run_job(VideoUpscaler(
                ffmpeg=self.ffmpeg, engine_factory=CrashingEngine,
            ))
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("stage"), "failed")
            self.assertTrue(store.get_metadata("extraction_complete"))
            self.assertEqual(store.frame_count("source"), 5)
        self.run_job()
        self.assertEqual(self.ffmpeg.extractions, 1)
        self.assertEqual(self.ffmpeg.encoded_values, [20, 40, 60, 80, 100])

    def test_default_spawn_workers_reuse_engines_without_gui_or_parent_session(self):
        logs = self.directory / "worker logs"
        logs.mkdir()
        with patch.dict(os.environ, {"QUALITYSCALER_TEST_WORKER_LOG": str(logs)}):
            self.run_job(VideoUpscaler(ffmpeg=self.ffmpeg, engine_factory=SpawnEngine))
        records = {
            int(path.stem): [json.loads(line) for line in path.read_text().splitlines()]
            for path in logs.glob("*.jsonl")
        }
        self.assertEqual(len(records), self.settings.workers)
        self.assertNotIn(os.getpid(), records)
        inferred = 0
        for events in records.values():
            initializations = [event for event in events if event["event"] == "init"]
            self.assertEqual(len(initializations), 1)
            self.assertFalse(initializations[0]["gui_loaded"])
            calls = [event["calls"] for event in events if event["event"] == "infer"]
            self.assertEqual(calls, list(range(1, len(calls) + 1)))
            inferred += len(calls)
        self.assertEqual(inferred, 5)
        self.assertEqual(self.ffmpeg.encoded_values, [20, 40, 60, 80, 100])

    def test_success_removes_cache_only_when_keep_frames_is_off(self):
        self.run_job(settings=replace(self.settings, keep_frames=False))
        self.assertFalse(self.cache.exists())
        self.assertTrue(self.output.is_file())

    def test_cleanup_does_not_remove_unrelated_cache_directory_files(self):
        self.cache.parent.mkdir()
        unrelated = self.cache.parent / "notes.txt"
        unrelated.write_text("keep me")
        self.run_job(settings=replace(self.settings, keep_frames=False))
        self.assertEqual(unrelated.read_text(), "keep me")
        self.assertFalse(self.cache.exists())

    def test_cancel_before_start_does_not_extract_or_encode(self):
        self.stop.set()
        with self.assertRaises(UpscaleCancelled):
            self.run_job()
        self.assertEqual(self.ffmpeg.extractions, 0)
        self.assertEqual(self.ffmpeg.encodings, 0)

    def test_invalid_settings_and_frames_fail_clearly(self):
        for changes in ({"workers": 0}, {"input_resize_factor": 0},
                        {"output_resize_factor": float("nan")},
                        {"blending_factor": 2}, {"extension": ".txt"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.settings, **changes)
        with self.assertRaisesRegex(ValueError, "decodable"):
            decode_frame(b"not an image")


if __name__ == "__main__":
    unittest.main()
