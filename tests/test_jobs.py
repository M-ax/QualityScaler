import multiprocessing
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

import numpy as np

from qualityscaler.encoders import EncoderOptions
from qualityscaler.ffmpeg import UpscaleCancelled
from qualityscaler.jobs import check_if_file_is_video, upscale_image, upscale_orchestrator
from qualityscaler.status import CacheSettingsConflict, StageStatus, VideoProgress
from qualityscaler.video import CacheSettingsMismatch, upscale_video


class BatchJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "clip.mp4"
        self.source.write_bytes(b"video")

    def run_batch(self, stop=None, *, files=None, retry_from=None, clear_cache=False):
        queue = Queue()
        upscale_orchestrator(
            queue, stop or Event(), [str(self.source)] if files is None else files,
            "output", "Testx2", 1,
            1, 1, "Auto", 1024, 0, True, ".png", ".mp4", "x264",
            retry_from=retry_from, clear_cache=clear_cache,
        )
        return queue.get_nowait()

    @patch("qualityscaler.jobs.upscale_video")
    def test_batch_calls_modular_video_backend(self, upscale):
        self.assertEqual(self.run_batch(), "Completed")
        upscale.assert_called_once_with(
            unittest.mock.ANY, unittest.mock.ANY, str(self.source), 1, "output", "Testx2",
            0, 1, "Auto", 1, 1, 1024, ".mp4", "x264", True,
        )

    @patch("qualityscaler.jobs.upscale_video")
    def test_batch_forwards_encoder_options_to_video_adapter(self, upscale):
        queue = Queue()
        options = EncoderOptions(preset="slow", rate_control="CRF", crf=20)
        upscale_orchestrator(
            queue, Event(), [str(self.source)], "output", "Testx2", 1,
            1, 1, "Auto", 1024, 0, True, ".png", ".mp4", "x264", options,
        )
        self.assertIs(upscale.call_args.args[-1], options)
        self.assertEqual(queue.get_nowait(), "Completed")

    @patch("qualityscaler.jobs.upscale_video", side_effect=UpscaleCancelled)
    def test_cancelled_video_does_not_report_completed(self, upscale):
        self.assertEqual(self.run_batch(), "Stop")

    @patch("qualityscaler.jobs.upscale_video", side_effect=RuntimeError("broken"))
    def test_failure_does_not_report_completed(self, upscale):
        self.assertEqual(self.run_batch(), "Error broken")

    @patch("qualityscaler.jobs.upscale_video", side_effect=CacheSettingsMismatch("different AI"))
    def test_cache_settings_mismatch_reports_a_structured_terminal_conflict(self, upscale):
        self.assertEqual(self.run_batch(), CacheSettingsConflict("different AI", 1))
        upscale.assert_called_once()

    @patch("qualityscaler.jobs.upscale_video", side_effect=ValueError("different AI"))
    def test_other_errors_with_the_same_message_are_not_cache_conflicts(self, upscale):
        self.assertEqual(self.run_batch(), "Error different AI")

    @patch("qualityscaler.jobs.upscale_video")
    @patch("qualityscaler.jobs.upscale_image")
    @patch("qualityscaler.jobs.AI_upscale")
    def test_retry_skips_completed_files_and_keeps_original_batch_numbers(
        self, engine, image, video,
    ):
        skipped = self.source.with_name("completed-but-now-missing.png")
        later_image = self.source.with_name("later.png")
        later_video = self.source.with_name("later.mp4")
        later_image.write_bytes(b"image")
        later_video.write_bytes(b"video")
        files = [str(path) for path in (skipped, self.source, later_image, later_video)]
        self.assertEqual(self.run_batch(files=files, retry_from=2, clear_cache=True), "Completed")
        self.assertEqual(
            [call.args[2:4] for call in video.call_args_list],
            [(str(self.source), 2), (str(later_video), 4)],
        )
        self.assertEqual(
            [call.kwargs for call in video.call_args_list], [{"clear_cache": True}, {}],
        )
        image.assert_called_once()
        self.assertEqual(image.call_args.args[1:3], (str(later_image), 3))
        engine.assert_called_once_with("Testx2", "Auto", 1, 1024)

    @patch("qualityscaler.jobs.upscale_video")
    def test_retry_does_not_authorize_clearing_a_later_video_conflict(self, video):
        later = self.source.with_name("later.mp4")
        later.write_bytes(b"video")
        video.side_effect = [None, CacheSettingsMismatch("later conflict")]
        files = ["already-completed.mp4", str(self.source), str(later)]
        self.assertEqual(
            self.run_batch(files=files, retry_from=2, clear_cache=True),
            CacheSettingsConflict("later conflict", 3),
        )
        self.assertEqual(
            [call.kwargs for call in video.call_args_list], [{"clear_cache": True}, {}],
        )
        self.assertEqual([call.args[3] for call in video.call_args_list], [2, 3])

    @patch("qualityscaler.jobs.upscale_video")
    def test_batch_position_alone_does_not_authorize_cache_reset(self, video):
        files = ["completed.mp4", str(self.source)]
        self.assertEqual(self.run_batch(files=files, retry_from=2), "Completed")
        self.assertEqual(video.call_args.args[3], 2)
        self.assertEqual(video.call_args.kwargs, {})

    @patch("qualityscaler.jobs.upscale_video")
    def test_clear_permission_without_a_retry_target_does_not_clear_anything(self, video):
        self.assertEqual(self.run_batch(clear_cache=True), "Completed")
        self.assertEqual(video.call_args.kwargs, {})

    @patch("qualityscaler.jobs.upscale_video")
    def test_retry_file_number_must_be_in_the_selected_batch(self, video):
        for retry_from in (0, -1, 2, True, False, 1.0, "1"):
            with self.subTest(retry_from=retry_from):
                self.assertEqual(
                    self.run_batch(retry_from=retry_from),
                    "Error retry_from must be a 1-based file number within the selected batch",
                )
        self.assertEqual(
            self.run_batch(files=[], retry_from=1),
            "Error retry_from must be a 1-based file number within the selected batch",
        )
        video.assert_not_called()

    @patch("qualityscaler.jobs.upscale_video")
    def test_cancelled_retry_does_not_start_the_authorized_video(self, video):
        stop = Event()
        stop.set()
        self.assertEqual(self.run_batch(stop, retry_from=1, clear_cache=True), "Stop")
        video.assert_not_called()

    @patch("qualityscaler.jobs.upscale_video")
    def test_stop_before_batch_does_not_start_next_file(self, upscale):
        stop = Event()
        stop.set()
        self.assertEqual(self.run_batch(stop), "Stop")
        upscale.assert_not_called()

    def test_video_adapter_sends_full_snapshots_with_the_batch_file_number(self):
        queue = Queue()
        with patch("qualityscaler.video.VideoUpscaler") as backend:
            upscale_video(
                queue, Event(), str(self.source), 2, self.temp.name, "Testx2",
                0, 2, "Auto", 1, 1, 1024, ".mp4", "x264", True,
            )
        self.assertEqual(backend.return_value.run.call_args.kwargs, {})
        report = backend.call_args.kwargs["progress"]
        report(VideoProgress("Extracting video frames", 3, 1, 5, False))
        stages = (
            StageStatus("extracting", "Extracting complete", "complete", 5, 5),
            StageStatus("upscaling", "Upscaling complete", "complete", 5, 5),
            StageStatus("encoding", "Encoding 20%", "running", 1, 5),
        )
        report(VideoProgress("Encoding upscaled video 20%", 5, 5, 5, True, stages))
        self.assertEqual(queue.get_nowait(), VideoProgress(
            "2. Encoding upscaled video 20%", 5, 5, 5, True, stages,
        ))

    def test_video_adapter_forwards_explicit_cache_reset_authorization(self):
        with patch("qualityscaler.video.VideoUpscaler") as backend:
            upscale_video(
                Queue(), Event(), str(self.source), 2, self.temp.name, "Testx2",
                0, 2, "Auto", 1, 1, 1024, ".mp4", "x264", True, clear_cache=True,
            )
        self.assertEqual(backend.return_value.run.call_args.kwargs, {"clear_cache": True})
        self.assertEqual(backend.return_value.run.call_args.args[0], str(self.source))

    def test_video_extensions_are_case_insensitive(self):
        self.assertTrue(check_if_file_is_video("movie.M4V"))
        self.assertTrue(check_if_file_is_video("movie.MP4"))
        self.assertTrue(check_if_file_is_video("old movie.wmv"))
        self.assertTrue(check_if_file_is_video("old movie.WMV"))
        self.assertFalse(check_if_file_is_video("movie.wmv.txt"))
        self.assertFalse(check_if_file_is_video("photo.png"))

    def test_image_output_is_resized_before_sharpening(self):
        original = np.zeros((8, 12, 3), np.uint8)
        native = np.zeros((16, 24, 3), np.uint8)
        resized = np.zeros((12, 18, 3), np.uint8)
        events = []
        engine = Mock()
        engine.AI_orchestration.side_effect = lambda image: events.append("AI") or native
        with patch("qualityscaler.jobs.image_read", return_value=original), \
                patch("qualityscaler.jobs.resize_with_output_factor",
                      side_effect=lambda image, factor: events.append("resize") or resized) as resize, \
                patch("qualityscaler.jobs.sharpen_and_save",
                      side_effect=lambda *args: events.append("sharpen")) as sharpen:
            upscale_image(Queue(), "photo.png", 1, "output", engine, "Testx2",
                          ".png", 1, 0.75, 0.5)
        self.assertEqual(events, ["AI", "resize", "sharpen"])
        self.assertIs(resize.call_args.args[0], native)
        self.assertEqual(resize.call_args.args[1], 0.75)
        self.assertIs(sharpen.call_args.args[1], resized)
        self.assertEqual(sharpen.call_args.args[2:], (0.5, ".png"))

    def test_batch_reuses_the_image_session(self):
        images = [Path(self.temp.name) / name for name in ("first.png", "second.png")]
        for image in images:
            image.write_bytes(b"image")
        engine = Mock()
        engine.AI_orchestration.return_value = np.zeros((8, 12, 3), np.uint8)
        queue = Queue()
        with patch("qualityscaler.jobs.AI_upscale", return_value=engine) as constructor, \
                patch("qualityscaler.jobs.image_read",
                      return_value=np.zeros((8, 12, 3), np.uint8)), \
                patch("qualityscaler.jobs.image_write"):
            upscale_orchestrator(
                queue, Event(), [str(path) for path in images], self.temp.name,
                "Testx2", 1, 1, 1, "Auto", 1024, 0, True, ".png", ".mp4", "x264",
            )
        constructor.assert_called_once_with("Testx2", "Auto", 1, 1024)
        self.assertEqual(engine.AI_orchestration.call_count, 2)
        self.assertEqual(queue.get_nowait(), "Completed")

    def test_orchestrator_runs_in_a_spawned_process_without_gui_state(self):
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        stop = context.Event()
        stop.set()
        process = context.Process(target=upscale_orchestrator, args=(
            queue, stop, ["clip.mp4"], "output", "Testx2", 1,
            1, 1, "Auto", 1024, 0, True, ".png", ".mp4", "x264",
        ))
        try:
            process.start()
            self.assertEqual(queue.get(timeout=15), "Stop")
            process.join(timeout=15)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.kill()
                process.join()
            queue.close()
            queue.join_thread()


if __name__ == "__main__":
    unittest.main()
