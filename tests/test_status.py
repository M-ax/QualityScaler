from dataclasses import FrozenInstanceError
import multiprocessing
import pickle
from queue import Empty, Queue
import unittest

from qualityscaler.status import CacheSettingsConflict, VideoProgress, write_process_status


class ProcessStatusTests(unittest.TestCase):
    def test_latest_snapshot_retains_both_stages(self):
        queue = Queue()
        write_process_status(queue, VideoProgress("Extracting", 10, 3, 100, False))
        latest = VideoProgress("Upscaling", 12, 4, 100, False)
        write_process_status(queue, latest)
        self.assertEqual(queue.get_nowait(), latest)
        with self.assertRaises(Empty):
            queue.get_nowait()

    def test_terminal_status_replaces_progress_and_legacy_messages_stay_strings(self):
        queue = Queue()
        write_process_status(queue, VideoProgress("Upscaling", 12, 4, None, False))
        write_process_status(queue, "Stop")
        self.assertEqual(queue.get_nowait(), "Stop")
        write_process_status(queue, 42)
        self.assertEqual(queue.get_nowait(), "42")

    def test_snapshot_is_picklable_and_survives_the_gui_manager_queue(self):
        snapshot = VideoProgress("2. Upscaling video", 12, 4, 100, False)
        self.assertEqual(pickle.loads(pickle.dumps(snapshot)), snapshot)
        with multiprocessing.get_context("spawn").Manager() as manager:
            queue = manager.Queue()
            write_process_status(queue, snapshot)
            received = queue.get_nowait()
            self.assertIsInstance(received, VideoProgress)
            self.assertEqual(received, snapshot)
            self.assertEqual(str(received), "2. Upscaling video")

    def test_cache_conflict_replaces_progress_without_becoming_a_string(self):
        queue = Queue()
        write_process_status(queue, VideoProgress("Upscaling", 12, 4, None, False))
        conflict = CacheSettingsConflict("The existing cache has not been changed.", 2)
        write_process_status(queue, conflict)
        self.assertIs(queue.get_nowait(), conflict)
        with self.assertRaises(Empty):
            queue.get_nowait()
        with self.assertRaises(FrozenInstanceError):
            conflict.file_number = 3

    def test_cache_conflict_is_picklable_and_survives_the_gui_manager_queue(self):
        conflict = CacheSettingsConflict("Different cached processing settings", 3)
        self.assertEqual(pickle.loads(pickle.dumps(conflict)), conflict)
        with multiprocessing.get_context("spawn").Manager() as manager:
            queue = manager.Queue()
            write_process_status(queue, VideoProgress("Upscaling", 12, 4, 100, False))
            write_process_status(queue, conflict)
            received = queue.get_nowait()
            self.assertIsInstance(received, CacheSettingsConflict)
            self.assertEqual(received, conflict)
            self.assertEqual(str(received), conflict.message)


if __name__ == "__main__":
    unittest.main()
