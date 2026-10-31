import multiprocessing
from queue import Queue
from threading import Event
import unittest
from unittest.mock import patch

from qualityscaler.ffmpeg import UpscaleCancelled
from qualityscaler.jobs import check_if_file_is_video, upscale_orchestrator


class BatchJobTests(unittest.TestCase):
    def run_batch(self, stop=None):
        queue = Queue()
        upscale_orchestrator(
            queue, stop or Event(), ["clip.mp4"], "output", "Testx2", 1,
            1, 1, "Auto", 1024, 0, True, ".png", ".mp4", "x264",
        )
        return queue.get_nowait()

    @patch("qualityscaler.jobs.upscale_video")
    def test_batch_calls_modular_video_backend(self, upscale):
        self.assertEqual(self.run_batch(), "Completed")
        upscale.assert_called_once()

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
        self.assertFalse(check_if_file_is_video("photo.png"))

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
