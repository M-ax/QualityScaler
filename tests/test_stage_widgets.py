"""Three independent stage rows use explicit success, never inferred percentages."""

import unittest
from unittest.mock import Mock

from qualityscaler.status import StageStatus, VideoProgress
from qualityscaler.widgets import StatusBar


def stages(*updates):
    by_name = {stage.stage: stage for stage in updates}
    return tuple(by_name.get(name, StageStatus(name))
                 for name in ("extracting", "upscaling", "encoding"))


class StageRowTests(unittest.TestCase):
    def setUp(self):
        self.bar = object.__new__(StatusBar)
        self.bar.video_rows = Mock()
        self.bar.label = Mock()
        self.bar._queue_fit = Mock()
        self.bar._video_progress = None
        for name in ("extraction", "upscaling", "encoding"):
            for part in ("label", "bar", "check", "detail"):
                setattr(self.bar, f"{name}_{part}", Mock())

    def display(self, *updates):
        snapshot = VideoProgress("1. Processing video", 100, 25, 100, True,
                                 stages=stages(*updates))
        self.bar.set_video_progress(snapshot)
        return snapshot

    def text(self, name, part):
        return getattr(self.bar, f"{name}_{part}").configure.call_args.kwargs["text"]

    def test_three_waiting_rows_have_no_success_check(self):
        self.display()
        for name, title in (("extraction", "Extraction"), ("upscaling", "AI upscaling"),
                            ("encoding", "Encoding")):
            self.assertTrue(self.text(name, "label").startswith(f"{title}:"))
            self.assertTrue(self.text(name, "label").endswith("waiting"))
            self.assertEqual(self.text(name, "check"), "")
            getattr(self.bar, f"{name}_bar").set.assert_called_with(0)
        self.bar.video_rows.grid.assert_called_once_with()

    def test_each_stage_keeps_its_own_latest_message_and_timing(self):
        self.display(
            StageStatus("extracting", "Cached source frames", "complete", 100, 100,
                        remaining_seconds=0, eta="2026-11-01 10:00:00 PST"),
            StageStatus("upscaling", "Upscaling video 25% (25/100 saved)", "running",
                        25, 100, remaining_seconds=93784, eta="2026-11-02 12:03:04 PST"),
        )
        self.assertEqual(self.text("extraction", "check"), "✓")
        self.assertIn("Cached source frames | Finished (local):", self.text("extraction", "detail"))
        upscale = self.text("upscaling", "detail")
        self.assertTrue(upscale.startswith("Upscaling video 25%"))
        self.assertIn("Remaining: 1d 2h 3m 4s", upscale)
        self.assertIn("ETA (local): 2026-11-02 12:03:04 PST", upscale)
        self.assertNotIn("Cached source frames", upscale)
        self.assertEqual(self.text("upscaling", "check"), "")
        self.assertEqual(self.text("encoding", "check"), "")
        self.bar.upscaling_bar.set.assert_called_with(0.25)

    def test_encoding_at_one_hundred_percent_is_not_yet_complete(self):
        self.display(
            StageStatus("extracting", "Extracted", "complete", 100, 100),
            StageStatus("upscaling", "Upscaled", "complete", 100, 100),
            StageStatus("encoding", "Encoding upscaled video 100%", "running", 100, 100,
                        remaining_seconds=0, eta="2026-11-01 10:00:00 PST"),
        )
        self.bar.encoding_bar.set.assert_called_with(1)
        self.assertEqual(self.text("encoding", "check"), "")
        self.bar.set_video_status("finalizing")
        self.assertEqual(self.text("encoding", "check"), "")
        self.assertTrue(self.text("encoding", "label").endswith("finalizing"))
        self.assertEqual(self.text("extraction", "check"), "✓")
        self.assertEqual(self.text("upscaling", "check"), "✓")

    def test_concurrent_tasks_keep_distinct_latest_messages_and_estimates(self):
        extract = StageStatus("extracting", "Extracting 20 saved", "running", 20, 100,
                              estimated_total=True, remaining_seconds=120,
                              eta="2026-11-01 10:02:00 PST")
        upscale = StageStatus("upscaling", "Upscaling 10 saved", "running", 10, 100,
                              estimated_total=True, remaining_seconds=600,
                              eta="2026-11-01 10:10:00 PST")
        self.display(extract, upscale)
        old_extraction = self.text("extraction", "detail")
        self.assertIn("Remaining: 2m 0s", old_extraction)
        self.assertIn("ETA (local): 2026-11-01 10:02:00 PST", old_extraction)
        self.assertIn("Remaining: 10m 0s", self.text("upscaling", "detail"))
        self.display(
            extract, StageStatus("upscaling", "Upscaling 15 saved", "running", 15, 100,
                                 estimated_total=True, remaining_seconds=540,
                                 eta="2026-11-01 10:09:00 PST"),
        )
        self.assertEqual(self.text("extraction", "detail"), old_extraction)
        self.assertIn("Upscaling 15 saved", self.text("upscaling", "detail"))
        self.assertNotIn("Upscaling 10 saved", self.text("upscaling", "detail"))
        self.assertIn("Remaining: 9m 0s", self.text("upscaling", "detail"))
        self.assertIn("ETA (local): 2026-11-01 10:09:00 PST", self.text("upscaling", "detail"))
        self.assertEqual(self.text("extraction", "check"), "")
        self.assertEqual(self.text("upscaling", "check"), "")

    def test_legacy_equal_counters_are_not_proof_of_upscaling_success(self):
        self.bar.set_video_progress(VideoProgress("1. Video", 100, 100, 100, True))
        self.assertEqual(self.text("extraction", "check"), "✓")
        self.assertEqual(self.text("upscaling", "check"), "")
        for state in ("finalizing", "error"):
            self.bar.set_video_status(state)
            self.assertEqual(self.text("upscaling", "check"), "")
            self.assertEqual(self.text("encoding", "check"), "")
            self.assertTrue(self.text("upscaling", "label").endswith(state))
        self.bar.set_video_status("job finished")
        for name in ("extraction", "upscaling", "encoding"):
            self.assertEqual(self.text(name, "check"), "✓")

    def test_successful_encoding_retains_all_three_checkmarks(self):
        self.display(*(StageStatus(key, "Saved successfully", "complete", 100, 100,
                                  remaining_seconds=0, eta="2026-11-01 10:00:00 PST")
                       for key in ("extracting", "upscaling", "encoding")))
        for name in ("extraction", "upscaling", "encoding"):
            self.assertEqual(self.text(name, "check"), "✓")
            getattr(self.bar, f"{name}_bar").set.assert_called_with(1)
        self.bar.set_video_status("job finished")
        for name in ("extraction", "upscaling", "encoding"):
            self.assertEqual(self.text(name, "check"), "✓")

    def test_terminal_success_covers_a_skipped_final_queue_snapshot(self):
        self.display(
            StageStatus("extracting", "Extracted", "complete", 100, 100),
            StageStatus("upscaling", "Upscaled", "complete", 100, 100),
            StageStatus("encoding", "Encoding upscaled video 25%", "running", 25, 100),
        )
        self.assertEqual(self.text("encoding", "check"), "")
        self.bar.set_video_status("job finished")
        self.assertEqual(self.text("encoding", "check"), "✓")
        self.bar.encoding_bar.set.assert_called_with(1)
        self.assertIn("100 / 100 frames", self.text("encoding", "label"))
        self.assertEqual(self.text("encoding", "detail"), "Completed successfully")

    def test_stop_and_error_never_complete_unfinished_or_waiting_stages(self):
        for state in ("stopping", "stopped", "error"):
            with self.subTest(state=state):
                self.display(
                    StageStatus("extracting", "Extracted", "complete", 100, 100),
                    StageStatus("upscaling", "Upscaling video 25%", "running", 25, 100),
                )
                self.bar.set_video_status(state)
                self.assertEqual(self.text("extraction", "check"), "✓")
                self.assertEqual(self.text("upscaling", "check"), "")
                self.assertEqual(self.text("encoding", "check"), "")
                self.assertTrue(self.text("upscaling", "label").endswith(state))
                self.assertTrue(self.text("encoding", "label").endswith("waiting"))
                self.assertNotIn("ETA", self.text("upscaling", "detail"))

    def test_new_job_and_image_status_clear_all_old_stage_state(self):
        self.display(*(StageStatus(key, "Complete", "complete", 100, 100)
                       for key in ("extracting", "upscaling", "encoding")))
        self.display()
        for name in ("extraction", "upscaling", "encoding"):
            self.assertEqual(self.text(name, "check"), "")
            self.assertEqual(self.text(name, "detail"), "")
        self.bar.set_video_progress(None)
        self.bar.video_rows.grid_remove.assert_called_once_with()
        for name in ("extraction", "upscaling", "encoding"):
            for part in ("label", "check", "detail"):
                self.assertEqual(self.text(name, part), "")
            getattr(self.bar, f"{name}_bar").set.assert_called_with(0)

    def test_detail_wrapping_uses_logical_not_physical_monitor_pixels(self):
        self.display(StageStatus("encoding", "Encoding", "running", 1, 100))
        self.bar._layout_after_id = "pending"
        self.bar._height = 24
        self.bar.video_rows.winfo_width.return_value = 1000
        self.bar._reverse_widget_scaling = Mock(return_value=500)
        self.bar.label._textbox.count.return_value = (0,)
        for name in ("extraction", "upscaling", "encoding"):
            getattr(self.bar, f"{name}_detail").cget.return_value = 600
        self.bar._fit_message()
        self.bar._reverse_widget_scaling.assert_called_once_with(1000)
        for name in ("extraction", "upscaling", "encoding"):
            getattr(self.bar, f"{name}_detail").configure.assert_called_with(wraplength=488)


if __name__ == "__main__":
    unittest.main()
