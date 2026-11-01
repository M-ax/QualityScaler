from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import sleep
import sys
import unittest
from unittest.mock import Mock, patch
from weakref import WeakSet

import cv2
import numpy as np

from qualityscaler.encoders import EncoderOptions
from qualityscaler.ffmpeg import UpscaleCancelled
from qualityscaler.frame_store import FrameStore
from qualityscaler.job_lock import JobAlreadyRunning, JobLock
from qualityscaler.video import (
    CacheSettingsMismatch, CURRENT_JOB_VERSION, VideoSettings, VideoUpscaler,
    decode_frame, get_resume_progress,
    initialize_worker, process_frame, SOURCE_LAYOUT_SCALED,
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
        self.input_sizes = []

    def extract(self, source, fps, stop, save_frame, *, input_size=None):
        self.extractions += 1
        self.input_sizes.append(input_size)
        for index, data in enumerate(self.frames):
            if input_size is not None:
                image = cv2.resize(decode_frame(data), input_size, interpolation=cv2.INTER_AREA)
                data = cv2.imencode(".jpg", image)[1].tobytes()
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
        capture = Mock()

        def property_value(prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return len(self.ffmpeg.frames)
            height, width = decode_frame(self.ffmpeg.frames[0]).shape[:2]
            return {cv2.CAP_PROP_FRAME_WIDTH: width, cv2.CAP_PROP_FRAME_HEIGHT: height}[prop]

        capture.get.side_effect = property_value
        self.capture = capture
        patcher = patch("qualityscaler.video.cv2.VideoCapture", return_value=capture)
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

    def run_job(self, upscaler=None, settings=None, *, clear_cache=False):
        return (upscaler or self.upscaler()).run(
            self.source, self.output, self.cache, settings or self.settings,
            clear_cache=clear_cache,
        )

    def prepare_legacy_cache(self, settings, *, extraction_complete):
        """Write the untagged version-2 layout produced before FFmpeg scaling."""
        self.cache.parent.mkdir()
        stat = self.source.stat()
        with FrameStore(self.cache) as store:
            store.set_metadata("job_version", 2)
            store.set_metadata("source", {
                "path": str(self.source.resolve()), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
            store.set_metadata("settings", asdict(settings))
            store.set_metadata("processing_settings", settings.processing_settings())
            store.set_metadata("output_path", str(self.output))
            store.set_metadata("stage", "paused")
            store.set_metadata("fps", 24)
            count = 5 if extraction_complete else 2
            for index, data in enumerate(self.ffmpeg.frames[:count]):
                store.put_frame("source", index, data)
            if extraction_complete:
                store.set_metadata("extraction_complete", True)
                store.set_metadata("total_frames", count)
                store.put_frame("upscaled", 0, jpeg(20))

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
            self.assertEqual(store.get_metadata("source_frame_layout"), SOURCE_LAYOUT_SCALED)
            self.assertEqual(store.get_metadata("job_version"), 3)
            self.assertEqual(store.get_metadata("source_size"), [12, 8])
            self.assertEqual(store.get_metadata("input_size"), [12, 8])

    def test_all_video_stages_publish_their_own_timing(self):
        messages = []
        runner = self.upscaler(status=messages.append)
        self.run_job(runner)
        for stage in ("Extracting video frames", "Upscaling video", "Encoding upscaled video"):
            self.assertTrue(any(stage in message for message in messages))
        for message in messages:
            for line in message.splitlines():
                self.assertEqual(line.count("Remaining:"), 1)
                self.assertEqual(line.count("ETA (local):"), 1)
        upscale = [
            line for message in messages for line in message.splitlines()
            if line.startswith("Upscaling video")
        ]
        self.assertTrue(any("estimating..." not in line for line in upscale))
        self.assertIs(self.ffmpeg.encoding_options["progress"], runner.task_status)
        self.assertEqual([stage.state for stage in runner.task_status.stages], ["complete"] * 3)
        self.assertEqual(messages[-1].count("Complete"), 3)

    def test_interrupted_video_retains_latest_status_without_false_completion(self):
        messages = []

        def status(message):
            messages.append(message)
            if any(line.startswith("Upscaling video") for line in message.splitlines()):
                self.stop.set()

        runner = self.upscaler(status=status)
        with self.assertRaises(UpscaleCancelled):
            self.run_job(runner)
        self.assertIn("Stopped", messages[-1])
        self.assertEqual(runner.task_status.stages[1].state, "stopped")
        self.assertEqual(runner.task_status.stages[2].state, "waiting")

    def test_completed_extraction_and_upscaling_remain_visible_during_encoding(self):
        snapshots = []
        runner = self.upscaler(progress=snapshots.append)
        original_encode = self.ffmpeg.encode

        def encode(*args, **kwargs):
            extracted, upscaled, encoding = snapshots[-1].stages
            self.assertEqual(extracted.state, "complete")
            self.assertEqual(upscaled.state, "complete")
            self.assertEqual(encoding.state, "running")
            self.assertEqual((extracted.completed, extracted.total), (5, 5))
            self.assertEqual((upscaled.completed, upscaled.total), (5, 5))
            self.assertEqual((extracted.remaining_seconds, upscaled.remaining_seconds), (0, 0))
            self.assertTrue(extracted.eta)
            self.assertTrue(upscaled.eta)
            finished = (extracted, upscaled)
            kwargs["progress"].update("encoding", "Encoding 100% (finalizing)", completed=5)
            self.assertEqual(snapshots[-1].stages[:2], finished)
            self.assertEqual(snapshots[-1].stages[2].state, "running")
            original_encode(*args, **kwargs)

        with patch.object(self.ffmpeg, "encode", side_effect=encode):
            self.run_job(runner)
        self.assertEqual([stage.state for stage in snapshots[-1].stages], ["complete"] * 3)
        self.assertEqual(snapshots[-1].stages[:2], snapshots[-2].stages[:2])

    def test_encoding_interruption_preserves_prior_success_but_not_encoding_success(self):
        for error in (RuntimeError("Encoder failed"), UpscaleCancelled()):
            with self.subTest(error=type(error).__name__):
                self.cache = self.directory / type(error).__name__ / "frames.sqlite3"
                snapshots = []
                runner = self.upscaler(progress=snapshots.append)

                def encode(*args, **kwargs):
                    kwargs["progress"].update("encoding", "Encoding 100% (finalizing)", completed=5)
                    raise error

                with patch.object(self.ffmpeg, "encode", side_effect=encode):
                    with self.assertRaises(type(error)):
                        self.run_job(runner)
                extracted, upscaled, encoding = snapshots[-1].stages
                self.assertEqual(extracted.state, "complete")
                self.assertEqual(upscaled.state, "complete")
                self.assertEqual(encoding.state, "stopped" if isinstance(error, UpscaleCancelled) else "error")
                self.assertEqual(encoding.completed, 5)
                self.assertIn("finalizing", encoding.message)

    def test_resumed_cache_restores_completed_stages_without_repeating_work(self):
        self.run_job()
        self.calls.clear()
        snapshots = []
        runner = self.upscaler(progress=snapshots.append)

        def encode(*args, **kwargs):
            extracted, upscaled, encoding = snapshots[-1].stages
            self.assertEqual(extracted.state, "complete")
            self.assertEqual(upscaled.state, "complete")
            self.assertIn("cached", extracted.message)
            self.assertEqual((extracted.completed, upscaled.completed), (5, 5))
            self.assertEqual(encoding.state, "running")

        with patch.object(self.ffmpeg, "encode", side_effect=encode):
            self.run_job(runner)
        self.assertEqual(self.ffmpeg.extractions, 1)
        self.assertEqual(self.calls, [])
        self.assertEqual([stage.state for stage in snapshots[-1].stages], ["complete"] * 3)

    def test_reused_runner_resets_all_stage_rows_and_counters_for_the_next_job(self):
        snapshots = []
        runner = self.upscaler(progress=snapshots.append)
        self.run_job(runner)
        snapshots.clear()
        self.cache = self.directory / "next-job" / "frames.sqlite3"
        self.run_job(runner)
        initial = snapshots[0]
        self.assertEqual((initial.extracted, initial.upscaled, initial.total), (0, 0, None))
        self.assertFalse(initial.extraction_complete)
        self.assertEqual([stage.state for stage in initial.stages], ["waiting"] * 3)
        self.assertTrue(all(stage.completed == 0 for stage in initial.stages))

    def test_decoder_failure_at_100_percent_does_not_complete_extraction(self):
        original_extract = self.ffmpeg.extract

        def extract(*args, **kwargs):
            original_extract(*args, **kwargs)
            raise RuntimeError("Decoder finalization failed")

        runner = self.upscaler()
        with patch.object(self.ffmpeg, "extract", side_effect=extract):
            with self.assertRaisesRegex(RuntimeError, "Decoder finalization failed"):
                self.run_job(runner)
        extraction = runner.task_status.stages[0]
        self.assertEqual(extraction.completed, 5)
        self.assertEqual(extraction.state, "error")
        self.assertTrue(extraction.estimated_total)
        self.assertEqual(self.ffmpeg.encodings, 0)

    def test_decoder_count_validation_failure_does_not_complete_extraction(self):
        original_extract = self.ffmpeg.extract

        def extract(*args, **kwargs):
            return original_extract(*args, **kwargs) + 1

        runner = self.upscaler()
        with patch.object(self.ffmpeg, "extract", side_effect=extract):
            with self.assertRaisesRegex(ValueError, "Extracted frame count does not match"):
                self.run_job(runner)
        self.assertEqual(runner.task_status.stages[0].state, "error")
        self.assertEqual(self.ffmpeg.encodings, 0)

    def test_upscaling_cleanup_failure_after_all_frames_never_completes_upscaling(self):
        class FailingShutdownExecutor(ThreadPoolExecutor):
            def shutdown(self, **kwargs):
                super().shutdown(**kwargs)
                raise RuntimeError("Worker cleanup failed")

        runner = self.upscaler(executor_factory=FailingShutdownExecutor)
        with self.assertRaisesRegex(RuntimeError, "Worker cleanup failed"):
            self.run_job(runner)
        extracted, upscaled, encoding = runner.task_status.stages
        self.assertEqual(extracted.state, "complete")
        self.assertEqual(upscaled.completed, 5)
        self.assertEqual(upscaled.state, "error")
        self.assertEqual(encoding.state, "waiting")
        self.assertEqual(self.ffmpeg.encodings, 0)

    def test_final_upscaling_commit_failure_does_not_complete_upscaling(self):
        cleaned_up = Event()
        original_commit = FrameStore.commit

        class ObservedExecutor(ThreadPoolExecutor):
            def shutdown(self, **kwargs):
                super().shutdown(**kwargs)
                cleaned_up.set()

        def commit(store):
            if cleaned_up.is_set():
                raise RuntimeError("Final commit failed")
            original_commit(store)

        runner = self.upscaler(executor_factory=ObservedExecutor)
        with patch.object(FrameStore, "commit", commit):
            with self.assertRaisesRegex(RuntimeError, "Final commit failed"):
                self.run_job(runner)
        extracted, upscaled, encoding = runner.task_status.stages
        self.assertEqual(extracted.state, "complete")
        self.assertEqual(upscaled.completed, 5)
        self.assertEqual(upscaled.state, "error")
        self.assertEqual(encoding.state, "waiting")
        self.assertEqual(self.ffmpeg.encodings, 0)

    def test_new_jobs_cache_input_scaled_frames_and_normalize_worker_factor(self):
        seen = []

        class RecordingEngine(FakeEngine):
            def AI_orchestration(inner, image):
                seen.append(image.shape)
                return super().AI_orchestration(image)

        def engine(settings):
            self.assertEqual(settings.input_resize_factor, 1)
            return RecordingEngine(self.calls)

        pipeline = VideoUpscaler(
            ffmpeg=self.ffmpeg, engine_factory=engine, executor_factory=ThreadPoolExecutor,
        )
        settings = replace(self.settings, input_resize_factor=0.5)
        self.run_job(pipeline, settings)
        self.assertEqual(self.ffmpeg.input_sizes, [(6, 4)])
        self.assertEqual(seen, [(4, 6, 3)] * 5)
        self.assertEqual(self.ffmpeg.encoding_options["output_size"], (12, 8))
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(decode_frame(store.get_frame("source", 0)).shape, (4, 6, 3))
            self.assertEqual(store.get_metadata("settings")["input_resize_factor"], 0.5)
            self.assertEqual(store.get_metadata("processing_settings")["input_resize_factor"], 0.5)
            self.assertEqual(store.get_metadata("input_size"), [6, 4])

    def test_untagged_partial_extraction_resumes_without_mixing_frame_sizes(self):
        settings = replace(self.settings, input_resize_factor=0.5)
        self.prepare_legacy_cache(settings, extraction_complete=False)
        seen_factors = []

        def engine(worker_settings):
            seen_factors.append(worker_settings.input_resize_factor)
            return FakeEngine(self.calls)

        self.run_job(VideoUpscaler(
            ffmpeg=self.ffmpeg, engine_factory=engine, executor_factory=ThreadPoolExecutor,
        ), settings)
        self.assertEqual(self.ffmpeg.input_sizes, [None])
        self.assertTrue(seen_factors)
        self.assertEqual(set(seen_factors), {0.5})
        with FrameStore(self.cache, read_only=True) as store:
            self.assertIsNone(store.get_metadata("source_frame_layout"))
            self.assertEqual(store.get_metadata("job_version"), 2)
            self.assertEqual(list(store.iter_frames("source")), self.ffmpeg.frames)
            self.assertEqual(store.get_metadata("stage"), "complete")

    def test_untagged_completed_extraction_preserves_saved_upscaled_frames(self):
        settings = replace(self.settings, input_resize_factor=0.5)
        self.prepare_legacy_cache(settings, extraction_complete=True)
        saved = jpeg(20)
        self.run_job(settings=settings)
        self.assertEqual(self.ffmpeg.extractions, 0)
        self.assertCountEqual(self.calls, [40, 60, 80, 100])
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.get_frame("upscaled", 0), saved)
            self.assertEqual(list(store.iter_frames("source")), self.ffmpeg.frames)
        self.assertEqual(get_resume_progress(self.source, str(self.directory), settings), 100)

    def test_unknown_source_layout_is_rejected_without_changing_the_job(self):
        self.run_job()
        with FrameStore(self.cache) as store:
            store.set_metadata("source_frame_layout", "unknown-resizer")
        before = self.cache.read_bytes()
        for clear_cache in (False, True):
            with self.subTest(clear_cache=clear_cache):
                with self.assertRaisesRegex(ValueError, "source frame layout") as raised:
                    self.run_job(
                        settings=replace(self.settings, blending_factor=0.5),
                        clear_cache=clear_cache,
                    )
                self.assertNotIsInstance(raised.exception, CacheSettingsMismatch)
                self.assertEqual(self.cache.read_bytes(), before)
        self.assertIsNone(get_resume_progress(
            self.source, str(self.directory), self.settings,
        ))

    def test_inconsistent_cache_version_and_layout_are_rejected_before_writes(self):
        self.run_job()
        for version, layout in ((2, SOURCE_LAYOUT_SCALED), (3, "original")):
            with self.subTest(version=version, layout=layout):
                with FrameStore(self.cache) as store:
                    store.set_metadata("job_version", version)
                    store.set_metadata("source_frame_layout", layout)
                before = self.cache.read_bytes()
                for clear_cache in (False, True):
                    with self.assertRaisesRegex(ValueError, "source frame layout") as raised:
                        self.run_job(
                            settings=replace(self.settings, blending_factor=0.5),
                            clear_cache=clear_cache,
                        )
                    self.assertNotIsInstance(raised.exception, CacheSettingsMismatch)
                    self.assertEqual(self.cache.read_bytes(), before)
                self.assertIsNone(get_resume_progress(
                    self.source, str(self.directory), self.settings,
                ))

    def test_unavailable_probe_dimensions_fall_back_to_a_decoded_frame(self):
        for index, dimensions in enumerate(((0, 0), (float("nan"), 8), (12, float("inf")))):
            with self.subTest(dimensions=dimensions):
                self.cache = self.directory / f"probe-{index}" / "frames.sqlite3"
                self.capture.get.side_effect = lambda prop: {
                    cv2.CAP_PROP_FRAME_COUNT: 5,
                    cv2.CAP_PROP_FRAME_WIDTH: dimensions[0],
                    cv2.CAP_PROP_FRAME_HEIGHT: dimensions[1],
                }[prop]
                self.capture.read.return_value = (True, decode_frame(self.ffmpeg.frames[0]))
                self.capture.reset_mock()
                self.run_job(settings=replace(self.settings, input_resize_factor=0.5))
                self.capture.read.assert_called_once()
                self.capture.release.assert_called_once()
                with FrameStore(self.cache, read_only=True) as store:
                    self.assertEqual(store.get_metadata("source_size"), [12, 8])
                    self.assertEqual(store.get_metadata("input_size"), [6, 4])

    def test_unreadable_dimension_fallback_releases_capture_and_saves_no_frames(self):
        self.capture.get.side_effect = lambda prop: 5 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
        self.capture.read.return_value = (False, None)
        with self.assertRaisesRegex(ValueError, "original video dimensions"):
            self.run_job()
        self.capture.release.assert_called_once()
        self.assertEqual(self.ffmpeg.extractions, 0)
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.frame_count("source"), 0)
            self.assertEqual(store.get_metadata("stage"), "failed")

    def test_partial_scaled_cache_rejects_changed_fallback_dimensions(self):
        def pause(message):
            if message.startswith("Extracting video frames") and "saved" in message:
                self.stop.set()

        with self.assertRaises(UpscaleCancelled):
            self.run_job(self.upscaler(status=pause))
        with FrameStore(self.cache, read_only=True) as store:
            before = list(store.iter_frames("source"))
        self.stop.clear()
        self.capture.get.side_effect = lambda prop: 5 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
        self.capture.read.return_value = (True, np.zeros((8, 14, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "source dimensions"):
            self.run_job()
        self.assertEqual(self.ffmpeg.extractions, 1)
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(list(store.iter_frames("source")), before)

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
            if any(line.startswith("Upscaling video") for line in message.splitlines()):
                self.stop.set()

        with self.assertRaises(UpscaleCancelled):
            self.run_job(self.upscaler(status=status))
        with FrameStore(self.cache, read_only=True) as store:
            saved = list(store.iter_indices("upscaled"))
            self.assertGreater(len(saved), 0)
            self.assertLess(len(saved), 5)
            self.assertEqual(store.get_metadata("stage"), "paused")
            extraction_complete = store.get_metadata("extraction_complete", False)
        self.assertFalse(self.output.exists())
        self.stop.clear()
        self.calls.clear()
        self.run_job()
        self.assertEqual(self.ffmpeg.extractions, 1 if extraction_complete else 2)
        self.assertCountEqual(self.calls, [
            value for index, value in enumerate((20, 40, 60, 80, 100))
            if index not in saved
        ])
        self.assertEqual(self.ffmpeg.encoded_values, [20, 40, 60, 80, 100])

    def test_partial_extraction_replays_and_reuses_committed_frames(self):
        def status(message):
            if message.startswith("Extracting video frames") and "(2 saved)" in message:
                self.stop.set()

        with self.assertRaises(UpscaleCancelled):
            self.run_job(self.upscaler(status=status))
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.frame_count("source"), 2)
            first_frame = store.get_frame("source", 0)
            self.assertFalse(store.get_metadata("extraction_complete", False))
        self.stop.clear()
        self.run_job()
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.get_frame("source", 0), first_frame)
            self.assertTrue(store.get_metadata("extraction_complete"))
        self.assertEqual(self.ffmpeg.extractions, 2)

    def test_upscaled_results_are_saved_while_extraction_waits_for_more_frames(self):
        first_saved = Event()
        extraction_finished = Event()
        snapshots = []
        original_extract = self.ffmpeg.extract

        def extract(source, fps, stop, save_frame, **options):
            def publish(index, data):
                save_frame(index, data)
                if index == 0:
                    self.assertTrue(first_saved.wait(5), "AI waited for extraction to finish")
            count = original_extract(source, fps, stop, publish, **options)
            extraction_finished.set()
            return count

        def progress(snapshot):
            snapshots.append(snapshot)
            if snapshot.upscaled and not first_saved.is_set():
                self.assertFalse(extraction_finished.is_set())
                self.assertFalse(snapshot.extraction_complete)
                self.assertEqual(snapshot.extracted, 1)
                lines = snapshot.message.splitlines()
                self.assertEqual(len(lines), 2)
                self.assertTrue(lines[0].startswith("Extracting video frames"))
                self.assertTrue(lines[1].startswith("Upscaling video"))
                for line in lines:
                    self.assertEqual(line.count("Remaining:"), 1)
                    self.assertEqual(line.count("ETA (local):"), 1)
                self.assertNotIn("Preparing video upscaling", snapshot.message)
                with FrameStore(self.cache, read_only=True) as store:
                    self.assertEqual(store.frame_count("upscaled"), 1)
                first_saved.set()

        self.ffmpeg.extract = extract
        self.run_job(self.upscaler(progress=progress))
        self.assertTrue(first_saved.is_set())
        self.assertTrue(extraction_finished.is_set())
        self.assertEqual(self.ffmpeg.encoded_values, [20, 40, 60, 80, 100])
        self.assertEqual((snapshots[-1].extracted, snapshots[-1].upscaled), (5, 5))
        self.assertTrue(snapshots[-1].extraction_complete)

    def test_last_frames_published_during_empty_queue_check_are_not_lost(self):
        timed_out = Event()
        producer_done = Event()
        self.ffmpeg.frames = self.ffmpeg.frames[:2]
        original_extract = self.ffmpeg.extract

        class RacingQueue(Queue):
            def get(inner, block=True, timeout=None):
                try:
                    return super().get(block=block, timeout=timeout)
                except Empty:
                    if timeout is not None and not timed_out.is_set():
                        timed_out.set()
                        self.assertTrue(producer_done.wait(5))
                    raise

        class ObservedProducer(ThreadPoolExecutor):
            def submit(inner, *args, **kwargs):
                future = super().submit(*args, **kwargs)
                future.add_done_callback(lambda result: producer_done.set())
                return future

        def extract(*args, **kwargs):
            self.assertTrue(timed_out.wait(5))
            return original_extract(*args, **kwargs)

        self.ffmpeg.extract = extract
        with patch("qualityscaler.video.Queue", RacingQueue), \
                patch("qualityscaler.video.ThreadPoolExecutor", ObservedProducer):
            self.run_job()
        self.assertTrue(producer_done.is_set())
        self.assertEqual(self.ffmpeg.encoded_values, [20, 40])

    def test_decoder_failure_after_saved_inference_never_encodes_and_resumes(self):
        first_saved = Event()
        original_extract = self.ffmpeg.extract

        def extract(source, fps, stop, save_frame, **options):
            save_frame(0, self.ffmpeg.frames[0])
            self.assertTrue(first_saved.wait(5))
            raise RuntimeError("Test decoder failed")

        self.ffmpeg.extract = extract
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            self.run_job(self.upscaler(
                progress=lambda snapshot: first_saved.set() if snapshot.upscaled else None,
            ))
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.frame_count("source"), 1)
            self.assertEqual(store.frame_count("upscaled"), 1)
            self.assertEqual(store.get_metadata("stage"), "failed")
            self.assertFalse(store.get_metadata("extraction_complete", False))
        self.assertFalse(self.stop.is_set(), "Internal failure must not become user cancellation")
        self.assertEqual(self.ffmpeg.encodings, 0)
        self.ffmpeg.extract = original_extract
        self.calls.clear()
        self.run_job()
        self.assertCountEqual(self.calls, [40, 60, 80, 100])
        self.assertEqual(self.ffmpeg.encoded_values, [20, 40, 60, 80, 100])

    def test_inference_failure_cancels_a_waiting_decoder_and_preserves_error(self):
        decoder_stopped = Event()

        def extract(source, fps, stop, save_frame, **options):
            try:
                save_frame(0, self.ffmpeg.frames[0])
                for _ in range(100):
                    if stop.is_set():
                        raise UpscaleCancelled()
                    sleep(0.01)
                self.fail("Decoder was not cancelled after inference failed")
            finally:
                decoder_stopped.set()

        self.ffmpeg.extract = extract
        pipeline = VideoUpscaler(
            ffmpeg=self.ffmpeg, stop_event=self.stop,
            engine_factory=lambda settings: FakeEngine(self.calls, fail_value=20),
            executor_factory=ThreadPoolExecutor,
        )
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_job(pipeline)
        self.assertTrue(decoder_stopped.is_set())
        self.assertFalse(self.stop.is_set())
        self.assertEqual(self.ffmpeg.encodings, 0)

    def test_full_extraction_queue_is_bounded_and_released_on_consumer_failure(self):
        queue_full = Event()
        decoder_stopped = Event()
        published = []
        capacity = 2 * self.settings.workers

        def extract(source, fps, stop, save_frame, **options):
            try:
                for index in range(1000):
                    if index == capacity + 1:
                        queue_full.set()
                    save_frame(index, self.ffmpeg.frames[0])
                    published.append(index)
                return 1000
            finally:
                decoder_stopped.set()

        def status(message):
            if "(1 saved)" in message:
                self.assertTrue(queue_full.wait(5))
                self.assertEqual(len(published), capacity + 1)
                raise RuntimeError("Test status consumer failed")

        self.ffmpeg.extract = extract
        with self.assertRaisesRegex(RuntimeError, "status consumer failed"):
            self.run_job(self.upscaler(status=status))
        self.assertTrue(decoder_stopped.is_set())
        self.assertEqual(len(published), capacity + 1)
        self.assertFalse(self.stop.is_set())
        self.assertEqual(self.ffmpeg.encodings, 0)
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.frame_count("source"), 1)
            self.assertEqual(store.get_metadata("stage"), "failed")

    def test_unknown_and_inaccurate_estimates_are_replaced_with_exact_total(self):
        original_get = self.capture.get.side_effect
        for expected in (0, float("nan"), 2, 20):
            with self.subTest(expected=expected):
                self.capture.get.side_effect = lambda prop: (
                    expected if prop == cv2.CAP_PROP_FRAME_COUNT else original_get(prop)
                )
                self.cache = self.directory / str(expected) / "frames.sqlite3"
                snapshots = []
                self.run_job(self.upscaler(progress=snapshots.append))
                extracting = next(
                    snapshot for snapshot in snapshots
                    if snapshot.stages[0].state == "running"
                )
                self.assertEqual(
                    extracting.total,
                    int(expected) if math.isfinite(expected) and expected > 0 else None,
                )
                self.assertEqual(snapshots[-1].total, 5)
                self.assertEqual(snapshots[-1].upscaled, 5)
                self.assertTrue(snapshots[-1].extraction_complete)

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
            if any(line.startswith("Upscaling video") for line in message.splitlines()):
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

            def get_metadata(self, key, default=None):
                return self.metadata.get(key, default)

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

    def test_encoder_choices_are_forwarded_saved_and_reuse_ai_frames(self):
        self.run_job()
        self.calls.clear()
        options = EncoderOptions(preset="veryfast", rate_control="CRF", crf=18.5)
        settings = replace(self.settings, encoder_options=options)
        self.run_job(settings=settings)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ffmpeg.extractions, 1)
        self.assertEqual(self.ffmpeg.encodings, 2)
        self.assertEqual(self.ffmpeg.encoding_options["encoder_options"], options)
        self.assertEqual(settings.processing_settings(), self.settings.processing_settings())
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("settings")["encoder_options"], asdict(options))

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
            if any(line.startswith("Upscaling video") for line in message.splitlines()):
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
        before = self.cache.read_bytes()
        with self.assertRaisesRegex(CacheSettingsMismatch, "different AI") as raised:
            self.run_job(settings=replace(self.settings, blending_factor=0.5))
        self.assertEqual(str(raised.exception), (
            "The cached job has different AI, resize, tile, blending, or sharpening settings. "
            "Restore its recorded settings or choose another output folder. "
            "The existing cache has not been changed."
        ))
        self.assertEqual(self.cache.read_bytes(), before)
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("stage"), "complete")
            self.assertEqual(store.get_metadata("settings")["blending_factor"], 0)
            self.assertEqual(store.frame_count("upscaled"), 5)

    def test_authorized_reset_regenerates_all_frames_and_metadata_without_deleting_files(self):
        self.prepare_legacy_cache(self.settings, extraction_complete=True)
        stale_metadata = {
            "stage": "failed", "error": "old failure", "interrupted_stage": "encoding",
            "fps": 99, "total_frames": 100, "extraction_complete": True,
            "source_size": [99, 99], "input_size": [88, 88], "updated_at": "old time",
            "unknown_old_metadata": {"must": "also disappear"},
        }
        with FrameStore(self.cache) as store:
            for key, value in stale_metadata.items():
                store.set_metadata(key, value)
            store.put_frame("source", 99, jpeg(200))
            store.put_frame("upscaled", 99, jpeg(200))
        note = self.cache.parent / "notes.txt"
        note.write_text("keep cache directory contents")
        other_output = self.directory / "another_upscaled.mp4"
        other_output.write_bytes(b"keep other output")
        self.output.write_bytes(b"previous output")
        settings = replace(self.settings, input_resize_factor=0.5)
        self.ffmpeg.frames = [jpeg(value) for value in (30, 50, 70)]
        pipeline = self.upscaler()
        extract = pipeline._extract
        reset = FrameStore.reset

        def checked_reset(store):
            with self.assertRaises(JobAlreadyRunning):
                with JobLock(self.cache):
                    pass
            return reset(store)

        def checked_extract(store, path, requested_settings):
            # Preparation must commit one fresh identity, not keep any state
            # from the old job (including unrecognized future metadata).
            with FrameStore(self.cache, read_only=True) as reader:
                self.assertEqual(reader.frame_count("source"), 0)
                self.assertEqual(reader.frame_count("upscaled"), 0)
                for key in stale_metadata:
                    self.assertIsNone(reader.get_metadata(key), key)
                self.assertEqual(reader.get_metadata("job_version"), CURRENT_JOB_VERSION)
                self.assertEqual(reader.get_metadata("source_frame_layout"), SOURCE_LAYOUT_SCALED)
                self.assertEqual(reader.get_metadata("processing_settings"), settings.processing_settings())
                self.assertEqual(reader.get_metadata("settings"), asdict(settings))
                self.assertEqual(reader.get_metadata("output_path"), str(self.output.resolve()))
            self.assertEqual(self.output.read_bytes(), b"previous output")
            return extract(store, path, requested_settings)

        with patch.object(FrameStore, "reset", autospec=True, side_effect=checked_reset) as reset_call, \
                patch.object(pipeline, "_extract", side_effect=checked_extract) as extract_call:
            self.run_job(pipeline, settings, clear_cache=True)
        reset_call.assert_called_once()
        extract_call.assert_called_once()
        self.assertEqual(self.ffmpeg.extractions, 1)
        self.assertEqual(self.ffmpeg.input_sizes, [(6, 4)])
        self.assertCountEqual(self.calls, [30, 50, 70])
        self.assertEqual(self.ffmpeg.encoded_values, [30, 50, 70])
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.frame_count("source"), 3)
            self.assertEqual(store.frame_count("upscaled"), 3)
            self.assertEqual(decode_frame(store.get_frame("source", 0)).shape, (4, 6, 3))
            self.assertEqual(store.get_metadata("total_frames"), 3)
            self.assertEqual(store.get_metadata("fps"), 24)
            self.assertEqual(store.get_metadata("source_size"), [12, 8])
            self.assertEqual(store.get_metadata("input_size"), [6, 4])
            self.assertEqual(store.get_metadata("stage"), "complete")
            self.assertIsNone(store.get_metadata("unknown_old_metadata"))
        self.assertEqual(note.read_text(), "keep cache directory contents")
        self.assertEqual(other_output.read_bytes(), b"keep other output")
        self.assertEqual(self.output.read_bytes(), b"encoded video")

    def test_clear_cache_with_matching_processing_settings_reuses_cache(self):
        self.run_job()
        with FrameStore(self.cache) as store:
            store.set_metadata("unknown_old_metadata", "keep when settings match")
            source_frames = list(store.iter_frames("source"))
            upscaled_frames = list(store.iter_frames("upscaled"))
        self.calls.clear()
        with patch.object(FrameStore, "reset", autospec=True) as reset:
            self.run_job(settings=replace(self.settings, codec="x265"), clear_cache=True)
        reset.assert_not_called()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ffmpeg.extractions, 1)
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(list(store.iter_frames("source")), source_frames)
            self.assertEqual(list(store.iter_frames("upscaled")), upscaled_frames)
            self.assertEqual(store.get_metadata("unknown_old_metadata"), "keep when settings match")
            self.assertEqual(store.get_metadata("settings")["codec"], "x265")

    def test_authorized_reset_and_new_identity_roll_back_together_on_preparation_failure(self):
        self.run_job()
        before = self.cache.read_bytes()
        output_before = self.output.read_bytes()
        settings = replace(self.settings, blending_factor=0.5)
        set_metadata = FrameStore.set_metadata

        def fail_preparation(store, key, value):
            set_metadata(store, key, value)
            if key == "output_path":
                with FrameStore(self.cache, read_only=True) as reader:
                    self.assertEqual(reader.frame_count("source"), 5)
                    self.assertEqual(reader.frame_count("upscaled"), 5)
                    self.assertEqual(reader.get_metadata("stage"), "complete")
                    self.assertEqual(reader.get_metadata("processing_settings"),
                                     self.settings.processing_settings())
                raise RuntimeError("Test preparation failure after reset")

        with patch.object(FrameStore, "set_metadata", autospec=True, side_effect=fail_preparation):
            with self.assertRaisesRegex(RuntimeError, "preparation failure"):
                self.run_job(settings=settings, clear_cache=True)
        self.assertEqual(self.cache.read_bytes(), before)
        self.assertEqual(self.output.read_bytes(), output_before)
        self.assertEqual(self.ffmpeg.extractions, 1)
        self.assertEqual(self.ffmpeg.encodings, 1)
        # The failed preparation must also release ownership for a later retry.
        self.run_job(settings=settings, clear_cache=True)
        self.assertEqual(self.ffmpeg.extractions, 2)

    def test_cancellation_before_authorized_reset_preserves_the_entire_old_job(self):
        self.run_job()
        before = self.cache.read_bytes()
        output_before = self.output.read_bytes()
        settings = replace(self.settings, blending_factor=0.5)
        for cancel_during_preparation in (False, True):
            with self.subTest(cancel_during_preparation=cancel_during_preparation):
                self.stop.clear()
                if not cancel_during_preparation:
                    self.stop.set()

                def digest(name):
                    self.stop.set()
                    return None

                with patch.object(FrameStore, "reset", autospec=True) as reset, \
                        patch("qualityscaler.video.model_digest", side_effect=digest):
                    with self.assertRaises(UpscaleCancelled):
                        self.run_job(settings=settings, clear_cache=True)
                reset.assert_not_called()
                self.assertEqual(self.cache.read_bytes(), before)
                self.assertEqual(self.output.read_bytes(), output_before)
                self.assertEqual(self.ffmpeg.extractions, 1)
                self.assertEqual(self.ffmpeg.encodings, 1)

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
        before = self.cache.read_bytes()
        for clear_cache in (False, True):
            with self.assertRaisesRegex(ValueError, "different processing version") as raised:
                self.run_job(
                    settings=replace(self.settings, blending_factor=0.5), clear_cache=clear_cache,
                )
            self.assertNotIsInstance(raised.exception, CacheSettingsMismatch)
            self.assertEqual(self.cache.read_bytes(), before)

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
        before = self.cache.read_bytes()
        for clear_cache in (False, True):
            with self.assertRaisesRegex(ValueError, "modified source") as raised:
                self.run_job(
                    settings=replace(self.settings, blending_factor=0.5), clear_cache=clear_cache,
                )
            self.assertNotIsInstance(raised.exception, CacheSettingsMismatch)
            self.assertEqual(self.cache.read_bytes(), before)
        with FrameStore(self.cache, read_only=True) as store:
            self.assertEqual(store.frame_count("source"), 5)
            self.assertEqual(store.get_metadata("stage"), "complete")

    def test_missing_source_identity_is_not_recoverable_by_authorizing_reset(self):
        self.run_job()
        with FrameStore(self.cache) as store:
            store.set_metadata("source", None)
        before = self.cache.read_bytes()
        for clear_cache in (False, True):
            with self.assertRaisesRegex(ValueError, "missing its job identity") as raised:
                self.run_job(
                    settings=replace(self.settings, blending_factor=0.5), clear_cache=clear_cache,
                )
            self.assertNotIsInstance(raised.exception, CacheSettingsMismatch)
            self.assertEqual(self.cache.read_bytes(), before)

    def test_competing_runner_is_rejected_before_modifying_metadata(self):
        self.run_job()
        before = self.cache.read_bytes()
        with JobLock(self.cache):
            with self.assertRaisesRegex(JobAlreadyRunning, "already running"):
                self.run_job(
                    settings=replace(self.settings, blending_factor=0.5), clear_cache=True,
                )
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
                        {"blending_factor": 2}, {"extension": ".txt"},
                        {"encoder_options": EncoderOptions(preset="p7")},
                        {"codec": "h264_nvenc", "encoder_options": EncoderOptions(rate_control="CRF")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.settings, **changes)
        with self.assertRaisesRegex(ValueError, "decodable"):
            decode_frame(b"not an image")


if __name__ == "__main__":
    unittest.main()
