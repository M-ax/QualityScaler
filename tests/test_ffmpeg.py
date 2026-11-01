from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
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

from qualityscaler.encoders import EncoderOptions
from qualityscaler.ffmpeg import FFmpeg, UpscaleCancelled
from qualityscaler.frame_store import FrameStore
from qualityscaler.status import PipelineStatus
from qualityscaler.video import CacheSettingsMismatch, VideoSettings, VideoUpscaler, decode_frame


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


class FFmpegStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "output.mp4"
        self.time = 0
        self.messages = []
        self.progress = PipelineStatus(self.messages.append, clock=lambda: self.time)

    def test_encoding_preserves_other_tasks_latest_status_and_own_timing(self):
        self.progress.start("upscaling", "Upscaling video 0%", total=100)
        self.time = 10
        self.progress.update("upscaling", "Upscaling video 20%", completed=20)

        @contextmanager
        def pipe(arguments, stop, *, writing, progress):
            self.assertEqual(arguments[arguments.index("-preset") + 1], "slow")
            self.assertEqual(arguments[arguments.index("-crf") + 1], "19.5")
            self.assertNotIn("-b:v", arguments)
            self.time = 12
            progress(2)
            yield BytesIO()

        ffmpeg = FFmpeg()
        with patch.object(ffmpeg, "_pipe", side_effect=pipe):
            ffmpeg.encode("source.mp4", self.output, 24, "x264", [b"jpeg"], Event(),
                          total_frames=4, progress=self.progress,
                          encoder_options=EncoderOptions("slow", "CRF", 19.5))
        message = next(message for message in self.messages if "Encoding upscaled video 50%" in message)
        lines = message.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("Upscaling video 20% | Remaining: 48s"))
        self.assertTrue(lines[1].startswith("Encoding upscaled video 50% | Remaining: 2s"))
        self.assertNotIn("Upscaling video 0%", message)
        self.assertEqual(message.count("ETA (local):"), 2)
        self.assertNotIn("Encoding", self.messages[-1])
        self.assertTrue(self.messages[-1].startswith("Upscaling video 20%"))

    def test_codec_retry_resets_timing_and_keeps_fallback_in_latest_status(self):
        attempts = []

        @contextmanager
        def pipe(arguments, stop, *, writing, progress):
            attempts.append(arguments)
            if len(attempts) == 1:
                self.time = 10
                progress(50)
                yield BytesIO()
                raise RuntimeError("Selected codec failed")
            self.time = 12
            progress(20)
            yield BytesIO()

        ffmpeg = FFmpeg()
        with patch.object(ffmpeg, "_pipe", side_effect=pipe):
            ffmpeg.encode("source.mp4", self.output, 24, "h264_nvenc", [b"jpeg"], Event(),
                          total_frames=100, progress=self.progress,
                          encoder_options=EncoderOptions(preset="p7", bitrate_kbps=4800))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0][attempts[0].index("-preset") + 1], "p7")
        self.assertNotIn("-preset", attempts[1])
        for arguments in attempts:
            self.assertEqual(arguments[arguments.index("-b:v") + 1], "4800k")
        fallback = [message for message in self.messages if "x264 fallback" in message]
        self.assertEqual(len(fallback), 2)
        self.assertIn("ETA (local): estimating...", fallback[0])
        self.assertIn("Encoding upscaled video (x264 fallback) 20% | Remaining: 8s", fallback[1])

    def test_unknown_encoding_total_reports_frames_without_a_fake_percentage(self):
        @contextmanager
        def pipe(arguments, stop, *, writing, progress):
            self.time = 10
            progress(5)
            yield BytesIO()

        ffmpeg = FFmpeg()
        with patch.object(ffmpeg, "_pipe", side_effect=pipe):
            ffmpeg.encode("source.mp4", self.output, 24, "x264", [b"jpeg"], Event(),
                          progress=self.progress)
        message = self.messages[-2]
        self.assertIn("Encoding upscaled video (5 frames)", message)
        self.assertIn("ETA (local): estimating...", message)
        self.assertNotIn("%", message)
        self.assertEqual(self.messages[-1], "")

    def test_failed_or_cancelled_encoding_clears_its_latest_status(self):
        for error in (RuntimeError("Encoder failed"), UpscaleCancelled()):
            with self.subTest(error=type(error).__name__):
                @contextmanager
                def pipe(arguments, stop, *, writing, progress):
                    self.time += 10
                    progress(5)
                    yield BytesIO()
                    raise error

                ffmpeg = FFmpeg()
                with patch.object(ffmpeg, "_pipe", side_effect=pipe):
                    with self.assertRaises(type(error)):
                        ffmpeg.encode("source.mp4", self.output, 24, "x264",
                                      iter([b"jpeg"]), Event(), total_frames=10,
                                      progress=self.progress)
                self.assertEqual(self.messages[-1], "")


class FFmpegEncoderOptionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mkv"
        self.output = self.root / "output.mp4"
        self.source.write_bytes(b"source video")
        self.ffmpeg = FFmpeg("ffmpeg")
        self.commands = []
        self.written_frames = []
        self.fail_attempts = set()
        pipe_patch = patch.object(self.ffmpeg, "_pipe", side_effect=self.fake_pipe)
        self.pipe = pipe_patch.start()
        self.addCleanup(pipe_patch.stop)

    @contextmanager
    def fake_pipe(self, arguments, stop_event, *, writing=False, progress=None):
        self.commands.append(list(arguments))
        self.assertTrue(writing)
        with BytesIO() as pipe:
            yield pipe
            self.written_frames.append(pipe.getvalue())
        if len(self.commands) in self.fail_attempts:
            raise RuntimeError("FFmpeg failed for this attempt")
        Path(arguments[-1]).write_bytes(b"encoded video")

    def encode(self, codec="x264", **kwargs):
        self.ffmpeg.encode(
            self.source, self.output, 5, codec, [b"jpeg1", b"jpeg2"], Event(), **kwargs,
        )

    def video_codec(self, command):
        # The first -c:v selects the MJPEG input decoder, the last the encoder.
        encoder_index = max(index for index, value in enumerate(command) if value == "-c:v")
        return command[encoder_index + 1]

    def test_default_and_explicit_none_preserve_fixed_bitrate_without_preset(self):
        for kwargs in ({}, {"encoder_options": None}, {"encoder_options": EncoderOptions()}):
            with self.subTest(kwargs=kwargs):
                self.encode(**kwargs)
                command = self.commands[-1]
                self.assertEqual(self.video_codec(command), "libx264")
                self.assertEqual(command[command.index("-b:v") + 1], "50000k")
                for flag in ("-preset", "-quality", "-crf"):
                    self.assertNotIn(flag, command)
                self.assertEqual(self.output.read_bytes(), b"encoded video")

    def test_software_crf_and_preset_replace_hardcoded_bitrate(self):
        for codec, canonical in (("x264", "libx264"), ("x265", "libx265"),
                                 ("libx264", "libx264"), ("libx265", "libx265")):
            with self.subTest(codec=codec):
                options = EncoderOptions("slow", "CRF", 18.25, -1)
                self.encode(codec, encoder_options=options)
                command = self.commands[-1]
                self.assertEqual(self.video_codec(command), canonical)
                self.assertEqual(command[command.index("-crf") + 1], "18.25")
                self.assertEqual(command[command.index("-preset") + 1], "slow")
                self.assertNotIn("-b:v", command)
                self.assertNotIn("-quality", command)

    def test_hardware_uses_its_own_preset_flag_and_selected_bitrate(self):
        for codec, preset, flag in (
            ("h264_nvenc", "p7", "-preset"), ("hevc_nvenc", "p1", "-preset"),
            ("h264_amf", "quality", "-quality"), ("hevc_amf", "speed", "-quality"),
            ("h264_qsv", "fast", "-preset"), ("hevc_qsv", "veryslow", "-preset"),
        ):
            with self.subTest(codec=codec):
                options = EncoderOptions(preset=preset, bitrate_kbps=6500, crf=float("nan"))
                self.encode(codec, encoder_options=options)
                command = self.commands[-1]
                self.assertEqual(self.video_codec(command), codec)
                self.assertEqual(command[command.index(flag) + 1], preset)
                self.assertEqual(command[command.index("-b:v") + 1], "6500k")
                self.assertNotIn("-crf", command)
                self.assertNotIn("-quality" if flag == "-preset" else "-preset", command)

    def test_invalid_primary_options_fail_before_tempfile_process_or_frame_iteration(self):
        self.output.write_bytes(b"previous output")
        for codec, options in (
            ("x264", EncoderOptions(preset="p7")),
            ("h264_nvenc", EncoderOptions(rate_control="CRF")),
            ("x265", EncoderOptions(rate_control="CRF", crf=float("nan"))),
            ("hevc_amf", EncoderOptions(bitrate_kbps=0)),
            ("x264", EncoderOptions(bitrate_kbps=1.5)),
        ):
            with self.subTest(codec=codec):
                with (
                    patch("qualityscaler.ffmpeg.tempfile.mkstemp") as mkstemp,
                    patch("qualityscaler.ffmpeg.subprocess.Popen") as popen,
                ):
                    with self.assertRaises(ValueError):
                        self.ffmpeg.encode(
                            self.source, self.output, 5, codec,
                            lambda: self.fail("Invalid options consumed frames"), Event(),
                            encoder_options=options,
                        )
                mkstemp.assert_not_called()
                popen.assert_not_called()
                self.pipe.assert_not_called()
                self.assertEqual(self.output.read_bytes(), b"previous output")
                self.assertEqual(self.source.read_bytes(), b"source video")
                self.assertFalse(any(self.root.glob(".*.partial-*")))

    def test_x264_fallback_drops_gpu_specific_presets_and_retains_bitrate(self):
        for codec, preset, expected_preset in (
            ("h264_nvenc", "p7", None), ("hevc_nvenc", "p1", None),
            ("h264_amf", "quality", None), ("hevc_amf", "speed", None),
            ("h264_qsv", "fast", "fast"), ("hevc_qsv", "veryslow", "veryslow"),
        ):
            with self.subTest(codec=codec):
                self.commands.clear()
                self.written_frames.clear()
                self.fail_attempts = {1}
                messages = []
                options = EncoderOptions(preset=preset, bitrate_kbps=4800)
                self.encode(codec, encoder_options=options, status=messages.append)
                self.assertEqual(len(self.commands), 2)
                fallback = self.commands[1]
                self.assertEqual(self.video_codec(fallback), "libx264")
                self.assertEqual(fallback[fallback.index("-b:v") + 1], "4800k")
                self.assertNotIn("-quality", fallback)
                if expected_preset is None:
                    self.assertNotIn("-preset", fallback)
                else:
                    self.assertEqual(fallback[fallback.index("-preset") + 1], expected_preset)
                self.assertEqual(self.written_frames, [b"jpeg1jpeg2"] * 2)
                self.assertTrue(any("x264 fallback" in message for message in messages))
                self.assertEqual(options.preset, preset)
                self.assertFalse(any(self.root.glob(".*.partial-*")))

    def test_software_fallback_retains_crf_and_shared_preset(self):
        self.fail_attempts = {1}
        self.encode("x265", encoder_options=EncoderOptions("slower", "CRF", 21.5, None))
        self.assertEqual([self.video_codec(command) for command in self.commands],
                         ["libx265", "libx264"])
        for command in self.commands:
            self.assertEqual(command[command.index("-crf") + 1], "21.5")
            self.assertEqual(command[command.index("-preset") + 1], "slower")
            self.assertNotIn("-b:v", command)

    def test_audio_retries_restore_primary_gpu_options_after_software_fallback(self):
        for codec, preset, flag in (
            ("h264_nvenc", "p6", "-preset"), ("hevc_amf", "balanced", "-quality"),
        ):
            with self.subTest(codec=codec):
                self.commands.clear()
                self.fail_attempts = {1, 2}
                options = EncoderOptions(preset=preset, bitrate_kbps=3600)
                self.encode(codec, encoder_options=options)
                self.assertEqual([self.video_codec(command) for command in self.commands],
                                 [codec, "libx264", codec])
                self.assertEqual(
                    [command[command.index("-c:a") + 1] for command in self.commands],
                    ["copy", "copy", "aac"],
                )
                self.assertEqual(self.commands[0][self.commands[0].index(flag) + 1], preset)
                self.assertEqual(self.commands[2][self.commands[2].index(flag) + 1], preset)
                self.assertNotIn("-preset", self.commands[1])
                self.assertNotIn("-quality", self.commands[1])
                for command in self.commands:
                    self.assertEqual(command[command.index("-b:v") + 1], "3600k")
                    self.assertNotIn("-crf", command)
                self.assertEqual(self.commands[2][self.commands[2].index("-b:a") + 1], "192k")

    def test_audio_retry_on_x264_keeps_crf_and_never_adds_video_bitrate(self):
        self.fail_attempts = {1}
        self.encode(encoder_options=EncoderOptions("fast", "CRF", 19.5, None))
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.commands[1][self.commands[1].index("-c:a") + 1], "aac")
        for command in self.commands:
            self.assertEqual(command[command.index("-crf") + 1], "19.5")
            self.assertEqual(command[command.index("-preset") + 1], "fast")
            self.assertNotIn("-b:v", command)

    def test_last_audio_fallback_uses_normalized_options_and_replays_frames(self):
        self.fail_attempts = {1, 2, 3}
        options = EncoderOptions(preset="quality", bitrate_kbps=8100)
        self.encode("h264_amf", encoder_options=options)
        self.assertEqual(len(self.commands), 4)
        self.assertEqual(self.video_codec(self.commands[-1]), "libx264")
        self.assertEqual(self.commands[-1][self.commands[-1].index("-c:a") + 1], "aac")
        self.assertNotIn("-quality", self.commands[-1])
        self.assertNotIn("-preset", self.commands[-1])
        for command in self.commands:
            self.assertEqual(command[command.index("-b:v") + 1], "8100k")
        self.assertEqual(self.written_frames, [b"jpeg1jpeg2"] * 4)

    def test_unknown_codec_keeps_legacy_x264_fallback(self):
        self.fail_attempts = {1}
        self.encode("not_a_video_codec")
        self.assertEqual([self.video_codec(command) for command in self.commands],
                         ["not_a_video_codec", "libx264"])

    def test_nonreplayable_frames_do_not_retry_or_replace_existing_output_on_failure(self):
        self.output.write_bytes(b"previous output")
        self.fail_attempts = {1}
        with self.assertRaisesRegex(RuntimeError, "FFmpeg failed"):
            self.ffmpeg.encode(
                self.source, self.output, 5, "h264_nvenc",
                iter([b"jpeg1"]), Event(), encoder_options=EncoderOptions(preset="p4"),
            )
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.output.read_bytes(), b"previous output")
        self.assertFalse(any(self.root.glob(".*.partial-*")))

    def test_failed_options_attempts_preserve_output_and_remove_partial(self):
        self.output.write_bytes(b"previous output")
        self.fail_attempts = {1, 2, 3, 4}
        with self.assertRaisesRegex(RuntimeError, "FFmpeg failed"):
            self.encode("h264_amf", encoder_options=EncoderOptions(preset="quality"))
        self.assertEqual(len(self.commands), 4)
        self.assertEqual(self.output.read_bytes(), b"previous output")
        self.assertEqual(self.source.read_bytes(), b"source video")
        self.assertFalse(any(self.root.glob(".*.partial-*")))

    def test_precancelled_encoding_does_not_validate_or_create_output(self):
        stop = Event()
        stop.set()
        with patch("qualityscaler.ffmpeg.tempfile.mkstemp") as mkstemp:
            with self.assertRaises(UpscaleCancelled):
                self.ffmpeg.encode(
                    self.source, self.output, 5, "x264", [], stop,
                    encoder_options=EncoderOptions(preset="invalid"),
                )
        mkstemp.assert_not_called()
        self.pipe.assert_not_called()
        self.assertFalse(self.output.exists())


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

    def make_wmv(self):
        source = self.root / "old family's vidéo.WMV"
        subprocess.run([
            self.ffmpeg.executable, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(self.source), "-c:v", "wmv2", "-c:a", "wmav2", str(source),
        ], check=True)
        return source

    def test_real_decoder_and_ai_overlap_before_encoding_with_audio(self):
        output = self.root / "streamed_upscaled.mp4"
        cache = self.root / "streamed cache" / "frames.sqlite3"
        first_saved = Event()
        extraction_finished = Event()
        original_extract = self.ffmpeg.extract

        def extract(source, fps, stop, save_frame, **options):
            def publish(index, data):
                save_frame(index, data)
                if index == 0:
                    self.assertTrue(first_saved.wait(5), "AI waited for the real decoder to finish")
            count = original_extract(source, fps, stop, publish, **options)
            extraction_finished.set()
            return count

        def progress(snapshot):
            if snapshot.upscaled and not first_saved.is_set():
                self.assertFalse(extraction_finished.is_set())
                self.assertFalse(snapshot.extraction_complete)
                with FrameStore(cache, read_only=True) as store:
                    self.assertEqual(store.frame_count("source"), 1)
                    self.assertEqual(store.frame_count("upscaled"), 1)
                first_saved.set()

        with patch.object(self.ffmpeg, "extract", side_effect=extract):
            VideoUpscaler(
                ffmpeg=self.ffmpeg, engine_factory=lambda settings: ResizeEngine(),
                executor_factory=ThreadPoolExecutor, checkpoint_interval=1, progress=progress,
            ).run(self.source, output, cache, VideoSettings(
                "FakeTestx2", keep_frames=True,
                encoder_options=EncoderOptions("veryfast", "CRF", 18.5),
            ))
        self.assertTrue(first_saved.is_set())
        self.assertTrue(extraction_finished.is_set())
        capture = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 5)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 64)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 48)
        finally:
            capture.release()
        subprocess.run([
            self.ffmpeg.executable, "-v", "error", "-i", str(output),
            "-map", "0:a:0", "-f", "null", "-",
        ], check=True)

    def test_settings_conflict_can_clear_only_its_cache_and_reencode_with_audio(self):
        output = self.root / "reset_upscaled.mp4"
        cache = self.root / "reset cache" / "frames.sqlite3"
        settings = VideoSettings("FakeTestx2", input_resize_factor=0.5, keep_frames=True)
        pipeline = VideoUpscaler(
            ffmpeg=self.ffmpeg, engine_factory=lambda settings: ResizeEngine(),
            executor_factory=ThreadPoolExecutor,
        )
        pipeline.run(self.source, output, cache, settings)
        previous_output = output.read_bytes()
        previous_cache = cache.read_bytes()
        changed = replace(settings, input_resize_factor=1.0)
        with self.assertRaises(CacheSettingsMismatch):
            pipeline.run(self.source, output, cache, changed)
        self.assertEqual(output.read_bytes(), previous_output)
        self.assertEqual(cache.read_bytes(), previous_cache)
        unrelated = cache.parent / "notes.txt"
        unrelated.write_text("Keep this file")
        original_extract = self.ffmpeg.extract

        def extract(*args, **kwargs):
            # Clearing cached frames must not remove an existing export.
            self.assertEqual(output.read_bytes(), previous_output)
            return original_extract(*args, **kwargs)

        with patch.object(self.ffmpeg, "extract", side_effect=extract) as extract:
            pipeline.run(self.source, output, cache, changed, clear_cache=True)
        extract.assert_called_once()
        self.assertEqual(unrelated.read_text(), "Keep this file")
        with FrameStore(cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("processing_settings"), changed.processing_settings())
            self.assertEqual(store.get_metadata("stage"), "complete")
            self.assertEqual(store.frame_count("source"), 5)
            self.assertEqual(store.frame_count("upscaled"), 5)
            self.assertEqual(decode_frame(store.get_frame("source", 0)).shape, (24, 32, 3))
        capture = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 5)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 64)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 48)
        finally:
            capture.release()
        subprocess.run([
            self.ffmpeg.executable, "-v", "error", "-i", str(output),
            "-map", "0:a:0", "-f", "null", "-",
        ], check=True)

    def test_software_presets_and_rate_modes_reencode_cached_frames(self):
        output = self.root / "configured_upscaled.mp4"
        cache = self.root / "configured cache" / "frames.sqlite3"
        processed = []

        class RecordingEngine(ResizeEngine):
            def AI_orchestration(inner, image):
                processed.append(image.shape)
                return super().AI_orchestration(image)

        pipeline = VideoUpscaler(
            ffmpeg=self.ffmpeg, engine_factory=lambda settings: RecordingEngine(),
            executor_factory=ThreadPoolExecutor,
        )
        for codec in ("x264", "x265"):
            for mode in ("CRF", "Bitrate"):
                with self.subTest(codec=codec, mode=mode):
                    options = EncoderOptions(
                        preset="veryfast", rate_control=mode, crf=18.5, bitrate_kbps=2500,
                    )
                    settings = VideoSettings("FakeTestx2", codec=codec, keep_frames=True,
                                             encoder_options=options)
                    with patch.object(self.ffmpeg, "_pipe", wraps=self.ffmpeg._pipe) as pipe:
                        pipeline.run(self.source, output, cache, settings)
                    commands = [call.args[0] for call in pipe.call_args_list
                                if call.kwargs.get("writing")]
                    # Verify the requested encoder accepted the options, rather
                    # than masking a failure with the automatic x264 fallback.
                    self.assertEqual(len(commands), 1)
                    self.assertIn("lib" + codec, commands[0])
                    self.assertIn("-crf" if mode == "CRF" else "-b:v", commands[0])
                    capture = cv2.VideoCapture(str(output))
                    try:
                        self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 5)
                        self.assertTrue(capture.read()[0])
                    finally:
                        capture.release()
                    self.assertEqual(len(processed), 5)  # No repeated AI work.
                    with FrameStore(cache, read_only=True) as store:
                        self.assertEqual(store.get_metadata("stage"), "complete")
                        self.assertEqual(store.get_metadata("settings")["encoder_options"]["rate_control"],
                                         mode)

    def test_scaled_extraction_resumes_without_resampling_inputs_twice(self):
        output = self.root / "scaled_upscaled.mp4"
        cache = self.root / "scaled cache" / "frames.sqlite3"
        settings = VideoSettings("FakeTestx2", input_resize_factor=0.5,
                                 workers=2, keep_frames=True)
        stop = Event()
        observed_shapes = []

        class NativeScaleEngine:
            def AI_orchestration(inner, image):
                observed_shapes.append(image.shape)
                return cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))

        def engine(worker_settings):
            self.assertEqual(worker_settings.input_resize_factor, 1)
            return NativeScaleEngine()

        def pause(message):
            if message.startswith("Extracting video frames") and "saved" in message:
                stop.set()

        pipeline = VideoUpscaler(
            ffmpeg=self.ffmpeg, engine_factory=engine, executor_factory=ThreadPoolExecutor,
            checkpoint_interval=1, stop_event=stop, status=pause,
        )
        with self.assertRaises(UpscaleCancelled):
            pipeline.run(self.source, output, cache, settings)
        with FrameStore(cache, read_only=True) as store:
            self.assertEqual(store.frame_count("source"), 1)
            first = store.get_frame("source", 0)
            self.assertEqual(decode_frame(first).shape, (12, 16, 3))
            self.assertEqual(store.get_metadata("source_size"), [32, 24])
            self.assertEqual(store.get_metadata("input_size"), [16, 12])
            self.assertFalse(store.get_metadata("extraction_complete", False))
        stop.clear()
        pipeline.status = lambda message: None
        pipeline.run(self.source, output, cache, settings)
        self.assertEqual(observed_shapes, [(12, 16, 3)] * 5)
        with FrameStore(cache, read_only=True) as store:
            self.assertEqual(store.get_frame("source", 0), first)
            self.assertEqual(store.get_metadata("stage"), "complete")
            self.assertEqual(store.get_metadata("settings")["input_resize_factor"], 0.5)
            self.assertEqual(decode_frame(store.get_frame("upscaled", 0)).shape, (24, 32, 3))
        capture = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 5)
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 5)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 32)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 24)
        finally:
            capture.release()
        subprocess.run([
            self.ffmpeg.executable, "-v", "error", "-i", str(output),
            "-map", "0:a:0", "-f", "null", "-",
        ], check=True)

    def test_scaling_preserves_display_geometry_for_rotated_odd_sized_video(self):
        original = self.root / "odd.mp4"
        rotated = self.root / "rotated.mp4"
        subprocess.run([
            self.ffmpeg.executable, "-y", "-v", "error", "-f", "lavfi",
            "-i", "testsrc=size=35x27:rate=5:duration=1", "-c:v", "libx264",
            "-pix_fmt", "yuv444p", str(original),
        ], check=True)
        rotation = subprocess.run([
            self.ffmpeg.executable, "-y", "-v", "error", "-display_rotation", "90",
            "-i", str(original), "-c", "copy", str(rotated),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if rotation.returncode:
            # Older FFmpeg writes the matrix via metadata; recent FFmpeg uses
            # the dedicated input option and may ignore the old rotate tag.
            subprocess.run([
                self.ffmpeg.executable, "-y", "-v", "error", "-i", str(original),
                "-c", "copy", "-metadata:s:v:0", "rotate=90", str(rotated),
            ], check=True)
        legacy_frames = []
        self.ffmpeg.extract(rotated, 5, Event(), lambda index, data: legacy_frames.append(data))
        self.assertEqual(decode_frame(legacy_frames[0]).shape, (35, 27, 3))

        class NativeScaleEngine:
            def AI_orchestration(inner, image):
                self.assertEqual(image.shape, (18, 14, 3))
                return cv2.resize(image, (28, 36))

        output = self.root / "rotated_upscaled.mp4"
        cache = self.root / "rotated cache" / "frames.sqlite3"
        settings = VideoSettings("FakeTestx2", input_resize_factor=0.5,
                                 output_resize_factor=1.25, keep_frames=True)
        VideoUpscaler(
            ffmpeg=self.ffmpeg, engine_factory=lambda settings: NativeScaleEngine(),
            executor_factory=ThreadPoolExecutor,
        ).run(rotated, output, cache, settings)
        with FrameStore(cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("source_size"), [27, 35])
            self.assertEqual(store.get_metadata("input_size"), [14, 18])
            self.assertEqual(decode_frame(store.get_frame("source", 0)).shape, (18, 14, 3))
        capture = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 36)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 46)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 5)
        finally:
            capture.release()

    def test_wmv_with_wma_audio_pauses_resumes_and_exports_mp4_with_aac(self):
        source = self.make_wmv()
        output = self.root / "wmv_upscaled.mp4"
        cache = self.root / "wmv cache" / "frames.sqlite3"
        settings = VideoSettings("FakeTestx2", workers=2, keep_frames=True)
        stop = Event()

        def pause(message):
            if any(line.startswith("Upscaling video") for line in message.splitlines()):
                stop.set()

        pipeline = VideoUpscaler(
            ffmpeg=self.ffmpeg, engine_factory=lambda settings: ResizeEngine(),
            executor_factory=ThreadPoolExecutor, stop_event=stop, status=pause,
        )
        with self.assertRaises(UpscaleCancelled):
            pipeline.run(source, output, cache, settings)
        with FrameStore(cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("stage"), "paused")
            self.assertGreater(store.frame_count("source"), 0)
            self.assertLessEqual(store.frame_count("source"), 5)
            extraction_complete = store.get_metadata("extraction_complete", False)
        stop.clear()
        messages = []
        pipeline.status = messages.append
        with patch.object(self.ffmpeg, "extract", wraps=self.ffmpeg.extract) as extract:
            pipeline.run(source, output, cache, settings)
        self.assertEqual(extract.call_count, 0 if extraction_complete else 1)
        self.assertTrue(any("AAC audio fallback" in message for message in messages))
        capture = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 5)
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 5)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 64)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 48)
            self.assertTrue(capture.read()[0])
        finally:
            capture.release()
        # ADTS accepts AAC, not WMA: this checks a real container-compatible
        # audio conversion, not merely the existence of an output stream.
        audio = subprocess.run([
            self.ffmpeg.executable, "-v", "error", "-i", str(output),
            "-map", "0:a:0", "-c:a", "copy", "-f", "adts", "-",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        self.assertGreater(len(audio), 7)
        self.assertEqual(audio[0], 0xFF)
        self.assertEqual(audio[1] & 0xF0, 0xF0)
        with FrameStore(cache, read_only=True) as store:
            self.assertEqual(store.get_metadata("stage"), "complete")
            self.assertEqual(store.frame_count("upscaled"), 5)

    def test_wmv_audio_still_passes_through_when_container_supports_wma(self):
        source = self.make_wmv()
        frames = []
        self.ffmpeg.extract(source, 5, Event(), lambda index, data: frames.append(data))
        output = self.root / "wmv_upscaled.mkv"
        messages = []
        with patch.object(self.ffmpeg, "_pipe", wraps=self.ffmpeg._pipe) as pipe:
            self.ffmpeg.encode(source, output, 5, "x264", frames, Event(),
                               total_frames=5, status=messages.append)
        self.assertEqual(pipe.call_count, 1)
        arguments = pipe.call_args.args[0]
        self.assertEqual(arguments[arguments.index("-c:a") + 1], "copy")
        self.assertFalse(any("AAC audio fallback" in message for message in messages))
        subprocess.run([
            self.ffmpeg.executable, "-v", "error", "-i", str(output),
            "-map", "0:a:0", "-f", "null", "-",
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
