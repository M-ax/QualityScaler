"""Progress snapshots are complete, synchronized, and independent of the GUI."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import multiprocessing
import pickle
from queue import Empty, Queue
from threading import Lock
from time import sleep
import unittest

from qualityscaler.status import (
    CacheSettingsConflict, PipelineStatus, StageStatus, VideoProgress,
    format_duration, write_process_status,
)


class DurationTests(unittest.TestCase):
    def test_human_readable_units_and_rounding(self):
        for seconds, expected in (
            (-2, "0s"), (0, "0s"), (0.1, "1s"), (59, "59s"),
            (59.1, "1m 0s"), (60, "1m 0s"), (3599, "59m 59s"),
            (3600, "1h 0m 0s"), (86399, "23h 59m 59s"),
            (86400, "1d 0h 0m 0s"), (93784, "1d 2h 3m 4s"),
        ):
            with self.subTest(seconds=seconds):
                self.assertEqual(format_duration(seconds), expected)


class PipelineStatusTests(unittest.TestCase):
    def setUp(self):
        self.time = 0.0
        self.now = datetime(2026, 1, 2, 23, 59, 50).astimezone().astimezone(timezone.utc)
        self.messages = []
        self.progress = PipelineStatus(
            self.messages.append, clock=lambda: self.time, now=lambda: self.now,
        )

    def test_parallel_stages_keep_only_their_latest_message_and_own_estimate(self):
        self.progress.start("extract", "Extracting 0/100", total=100)
        self.progress.start("upscale", "Upscaling 0/100", total=100)
        self.time = 10
        self.progress.update("extract", "Extracting 50/100", completed=50)
        self.progress.update("upscale", "Upscaling 10/100", completed=10)
        message = self.messages[-1]
        lines = message.splitlines()
        self.assertEqual(len(lines), 2)
        for line, label, seconds in (
            (lines[0], "Extracting 50/100", 10),
            (lines[1], "Upscaling 10/100", 90),
        ):
            self.assertTrue(line.startswith(f"{label} | Remaining: {format_duration(seconds)}"))
            self.assertEqual(line.count("Remaining:"), 1)
            self.assertEqual(line.count("ETA (local):"), 1)
            expected_eta = (self.now + timedelta(seconds=seconds)).astimezone()
            self.assertIn(f"ETA (local): {expected_eta:%Y-%m-%d %H:%M:%S %Z}", line)
        self.assertNotIn("Extracting 0/100", message)
        self.assertNotIn("Upscaling 0/100", message)

    def test_unknown_rate_or_total_does_not_invent_an_eta(self):
        self.progress.start("upscale", "Loading model", total=100)
        self.assertIn("Remaining: estimating...", self.messages[-1])
        self.assertIn("ETA (local): estimating...", self.messages[-1])
        self.time = 10
        self.progress.update("upscale", "Upscaling 10%", completed=10)
        self.assertIn("Remaining: 1m 30s", self.messages[-1])
        self.progress.start("extract", "Extracting", total=None)
        self.progress.update("extract", "Extracting (10 saved)", completed=10)
        lines = self.messages[-1].splitlines()
        self.assertIn("Remaining: 1m 30s", lines[0])
        self.assertIn("ETA (local): estimating...", lines[1])
        self.progress.finish("extract")
        self.assertIn("Extracting (10 saved) | Remaining: 0s", self.messages[-1])
        self.assertIn("Complete", self.messages[-1])
        self.assertIn("Remaining: 1m 30s", self.messages[-1])

    def test_resume_uses_only_new_work_for_its_rate(self):
        self.progress.start("upscale", "Resuming 90/100", completed=90, total=100)
        self.time = 10
        self.progress.update("upscale", "Upscaling 92/100", completed=92)
        self.assertIn("Remaining: 40s", self.messages[-1])
        self.progress.update("upscale", "Out-of-order notification", completed=91)
        self.assertIn("Remaining: 40s", self.messages[-1])

    def test_exact_total_updates_without_resetting_the_latest_message_or_rate(self):
        self.progress.start("upscale", "Upscaling", total=None)
        self.time = 10
        self.progress.update("upscale", "Upscaling (10 saved)", completed=10)
        self.assertIn("ETA (local): estimating...", self.messages[-1])
        self.progress.update("upscale", total=30)
        self.assertTrue(self.messages[-1].startswith("Upscaling (10 saved) | Remaining: 20s"))

    def test_exhausted_probe_estimate_does_not_claim_completion_before_eof(self):
        self.progress.start("upscale", "Upscaling", total=10, estimated_total=True)
        self.time = 10
        for completed in (10, 20):
            self.progress.update("upscale", "Upscaling", completed=completed)
            self.assertIn("Remaining: estimating...", self.messages[-1])
            self.assertIn("ETA (local): estimating...", self.messages[-1])
        self.progress.update("upscale", total=30, estimated_total=False)
        self.assertIn("Remaining: 5s", self.messages[-1])

    def test_eta_uses_local_date_time_and_zone_across_midnight(self):
        self.progress.start("upscale", "Upscaling", total=3)
        self.time = 10
        self.progress.update("upscale", "Upscaling 1/3", completed=1)
        local_eta = (self.now + timedelta(seconds=20)).astimezone()
        self.assertEqual(local_eta.date().isoformat(), "2026-01-03")
        self.assertTrue(self.messages[-1].endswith(
            f"Remaining: 20s | ETA (local): {local_eta:%Y-%m-%d %H:%M:%S %Z}"
        ))

    def test_restart_resets_the_attempt_rate_and_progress(self):
        self.progress.start("encoding", "Encoding", total=100)
        self.time = 10
        self.progress.update("encoding", "Encoding 50%", completed=50)
        self.assertIn("Remaining: 10s", self.messages[-1])
        self.time = 100
        self.progress.start("encoding", "Encoding (x264 fallback)", total=100)
        self.assertIn("ETA (local): estimating...", self.messages[-1])
        self.time = 102
        self.progress.update("encoding", "Encoding (x264 fallback) 20%", completed=20)
        self.assertIn("Remaining: 8s", self.messages[-1])

    def test_finished_stage_is_retained_and_late_updates_are_ignored(self):
        self.progress.start("extracting", "Extracting", total=100)
        self.progress.start("upscaling", "Upscaling", total=100)
        self.time = 10
        self.progress.update("upscaling", "Upscaling 50%", completed=50)
        self.progress.finish("extracting", "Extracting 100%", completed=100)
        finished = self.progress.stages[0]
        self.assertEqual(finished.state, "complete")
        self.assertEqual(finished.remaining_seconds, 0)
        self.assertEqual(finished.eta, f"{self.now.astimezone():%Y-%m-%d %H:%M:%S %Z}")
        self.assertTrue(self.messages[-1].startswith("Extracting 100% |"))
        self.assertIn("Remaining: 10s", self.messages[-1])
        count = len(self.messages)
        self.progress.update("extracting", "Stale extraction", completed=99)
        self.assertEqual(len(self.messages), count)
        self.progress.finish("upscaling", completed=100)
        self.time = 500
        self.now += timedelta(seconds=500)
        self.progress.start("encoding", "Encoding", total=100)
        self.assertEqual(self.progress.stages[0], finished)
        self.assertEqual(self.progress.stages[1].state, "complete")
        self.assertEqual(self.progress.stages[2].state, "running")
        count = len(self.messages)
        self.progress.update("upscaling", "Stale inference", completed=99)
        self.progress.finish("upscaling")
        self.assertEqual(len(self.messages), count)

    def test_clear_resets_rows_for_a_new_job(self):
        self.progress.finish("extracting", "Cached extraction", completed=100, total=100)
        self.progress.start("upscaling", "Upscaling", total=100)
        self.progress.clear()
        self.assertEqual(self.messages[-1], "")
        self.assertEqual(self.progress.stages, tuple(StageStatus(stage) for stage in (
            "extracting", "upscaling", "encoding",
        )))
        count = len(self.messages)
        self.progress.update("extracting", "Stale extraction", completed=99)
        self.progress.update("upscaling", "Stale inference", completed=99)
        self.progress.clear()
        self.assertEqual(len(self.messages), count)

    def test_interrupt_freezes_latest_timing_without_completing_running_stages(self):
        for state in ("stopped", "error"):
            with self.subTest(state=state):
                self.progress.clear()
                self.progress.finish("extracting", "Cached extraction", completed=100, total=100)
                self.progress.start("upscaling", "Upscaling", total=100)
                self.time += 10
                self.progress.update("upscaling", "Upscaling 50%", completed=50)
                previous = self.progress.stages[1]
                self.progress.interrupt(state)
                finished, interrupted, waiting = self.progress.stages
                self.assertEqual(finished.state, "complete")
                self.assertEqual(interrupted.state, state)
                self.assertEqual(interrupted.message, previous.message)
                self.assertEqual(interrupted.remaining_seconds, previous.remaining_seconds)
                self.assertEqual(interrupted.eta, previous.eta)
                self.assertEqual(waiting.state, "waiting")
                snapshot = self.progress.stages
                self.time += 100
                self.progress.update("upscaling", "Stale callback", completed=100)
                self.progress.finish("upscaling")
                self.assertEqual(self.progress.stages, snapshot)

    def test_snapshots_are_immutable_and_available_inside_the_string_callback(self):
        received = []
        progress = PipelineStatus(lambda message: received.append((message, progress.stages)))
        self.assertEqual(len(progress.stages), 3)
        progress.start("extracting", "Extracting", total=10, estimated_total=True)
        progress.update("extracting", "Extracting 50%", completed=5)
        self.assertEqual(received[-1][1], progress.stages)
        self.assertEqual(received[0][1][0].completed, 0)
        self.assertEqual(received[-1][1][0].completed, 5)
        self.assertTrue(received[-1][1][0].estimated_total)
        with self.assertRaises(FrozenInstanceError):
            progress.stages[0].state = "complete"

    def test_completed_progress_has_no_negative_remaining_time(self):
        self.progress.start("encoding", "Encoding", total=10)
        self.time = 10
        self.progress.update("encoding", "Encoding 100%", completed=12)
        self.assertIn("Remaining: 0s", self.messages[-1])
        self.assertEqual(self.progress.stages[2].state, "running")

    def test_simultaneous_updates_serialize_publication_of_complete_snapshots(self):
        active = 0
        peak = 0
        counter_lock = Lock()
        messages = []

        def publish(message):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            sleep(0.001)
            messages.append(message)
            with counter_lock:
                active -= 1

        progress = PipelineStatus(publish)
        stages = ("Extracting", "Upscaling", "Encoding")
        for stage in stages:
            progress.start(stage, f"{stage} 0", total=20)

        def update(stage):
            for count in range(1, 21):
                progress.update(stage, f"{stage} {count}", completed=count)

        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(update, stages))
        self.assertEqual(peak, 1)
        for message in messages[2:]:
            for stage in stages:
                self.assertIn(stage, message)
            self.assertEqual(message.count("ETA (local):"), len(stages))
            self.assertEqual(len(message.splitlines()), len(stages))
        for stage in stages:
            self.assertIn(f"{stage} 20", messages[-1])

    def test_latest_status_queue_receives_a_whole_snapshot(self):
        queue = Queue(maxsize=1)
        progress = PipelineStatus(lambda message: write_process_status(queue, f"2. {message}"))
        progress.start("extract", "Extracting", total=100)
        progress.start("upscale", "Upscaling", total=100)
        progress.update("upscale", "Upscaling 10%", completed=10)
        message = queue.get_nowait()
        self.assertTrue(message.startswith("2. Extracting |"))
        self.assertIn("\nUpscaling 10% |", message)
        self.assertEqual(message.count("ETA (local):"), 2)
        self.assertTrue(queue.empty())


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
        snapshot = VideoProgress("2. Upscaling video", 12, 4, 100, False, (
            StageStatus("extracting", "Extracting", "running", 12, 100, True, 30, "local ETA"),
            StageStatus("upscaling", "Upscaling", "running", 4, 100),
            StageStatus("encoding"),
        ))
        self.assertEqual(pickle.loads(pickle.dumps(snapshot)), snapshot)
        with multiprocessing.get_context("spawn").Manager() as manager:
            queue = manager.Queue()
            write_process_status(queue, snapshot)
            received = queue.get_nowait()
            self.assertIsInstance(received, VideoProgress)
            self.assertEqual(received, snapshot)
            self.assertEqual(str(received), "2. Upscaling video")

    def test_old_positional_snapshot_constructor_keeps_default_stages(self):
        snapshot = VideoProgress("Extracting", 0, 0, None, False)
        self.assertEqual(snapshot.stages, ())

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
