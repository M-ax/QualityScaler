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


class BatchJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "clip.mp4"
        self.source.write_bytes(b"video")

    def run_batch(self, stop=None):
        queue = Queue()
        upscale_orchestrator(
            queue, stop or Event(), [str(self.source)], "output", "Testx2", 1,
            1, 1, "Auto", 1024, 0, True, ".png", ".mp4", "x264",
        )
        return queue.get_nowait()

    @patch("qualityscaler.jobs.upscale_video")
    def test_batch_calls_modular_video_backend(self, upscale):
        self.assertEqual(self.run_batch(), "Completed")
        upscale.assert_called_once()

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

    @patch("qualityscaler.jobs.upscale_video")
    def test_stop_before_batch_does_not_start_next_file(self, upscale):
        stop = Event()
        stop.set()
        self.assertEqual(self.run_batch(stop), "Stop")
        upscale.assert_not_called()

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
