"""Native layout checks for persistent stage rows and their success indicators."""

import os
import sys
import unittest

from qualityscaler import gui
from qualityscaler.status import StageStatus, VideoProgress
from qualityscaler.widgets import STAGE_COMPLETE_COLOR
import test_layout


@unittest.skipUnless(sys.platform == "win32" or os.environ.get("DISPLAY"),
                     "Stage layout tests need a desktop display")
class PersistentStageLayoutTests(unittest.TestCase):
    def run_desktop(self, check, *, system_dpi=None):
        test_layout.DesktopLayoutTests.run_desktop(self, check, system_dpi=system_dpi)

    def test_three_rows_and_checkmarks_survive_the_full_video_lifecycle(self):
        eta = "2026-11-02 12:03:04 Pacific Standard Time"
        frames = 100
        initial = tuple(StageStatus(key) for key in ("extracting", "upscaling", "encoding"))
        concurrent = (
            StageStatus("extracting", "Extracting video frames 50% (50 saved)", "running",
                        50, frames, True, 93784, eta),
            StageStatus("upscaling", "Upscaling video 25% (25/100 saved)", "running",
                        25, frames, True, 281352, eta),
            StageStatus("encoding"),
        )
        encoding = (
            StageStatus("extracting", "Extracting video frames 100% (100 saved)", "complete",
                        frames, frames, remaining_seconds=0, eta=eta),
            StageStatus("upscaling", "Upscaling video 100% (100/100 saved)", "complete",
                        frames, frames, remaining_seconds=0, eta=eta),
            StageStatus("encoding", "Encoding upscaled video 100%", "running", frames, frames,
                        remaining_seconds=0, eta=eta),
        )
        complete = (
            *encoding[:2],
            StageStatus("encoding", "Encoding upscaled video 100%", "complete", frames, frames,
                        remaining_seconds=0, eta=eta),
        )

        def check(window):
            state = gui.app_state
            gui.App.place_stop_button()
            window.geometry("1000x540")
            for stages in (initial, concurrent, encoding, complete, initial):
                source, upscale, _ = stages
                gui._display_status(VideoProgress(
                    "1. Processing video", source.completed, upscale.completed, frames,
                    source.state == "complete", stages=stages,
                ))
                window.update()
                bar = state.status_bar
                self.assertTrue(bar.video_rows.winfo_ismapped())
                previous_bottom = None
                for name, stage in zip(("extraction", "upscaling", "encoding"), stages):
                    label = getattr(bar, f"{name}_label")
                    progress = getattr(bar, f"{name}_bar")
                    indicator = getattr(bar, f"{name}_check")
                    detail = getattr(bar, f"{name}_detail")
                    with self.subTest(stage=stage.stage, state=stage.state):
                        self.assertTrue(label.winfo_ismapped())
                        self.assertTrue(progress.winfo_ismapped())
                        self.assertEqual(indicator.cget("text"),
                                         "✓" if stage.state == "complete" else "")
                        self.assertEqual(indicator.cget("text_color"), STAGE_COMPLETE_COLOR)
                        self.assertLessEqual(progress.winfo_rootx() + progress.winfo_width(),
                                             indicator.winfo_rootx())
                        if previous_bottom is not None:
                            self.assertLessEqual(previous_bottom, label.winfo_rooty())
                        previous_bottom = label.winfo_rooty() + label.winfo_height()
                        if detail.cget("text"):
                            self.assertTrue(detail.winfo_ismapped())
                            self.assertIn(stage.message, detail.cget("text"))
                            previous_bottom = detail.winfo_rooty() + detail.winfo_height()
                            self.assertLessEqual(previous_bottom,
                                                 bar.winfo_rooty() + bar.winfo_height())
                self.assertEqual(bar.winfo_y() + bar.winfo_height(), window.winfo_height())
                self.assertLessEqual(
                    state.action_button.winfo_rooty() + state.action_button.winfo_height(),
                    bar.winfo_rooty(),
                )
                self.assertEqual(bar.label.get("1.0", "end-1c"), "Processing video")
            gui._display_status("2. Upscaling image")
            window.update()
            self.assertFalse(state.status_bar.video_rows.winfo_ismapped())

        for dpi in (1, 1.25, 1.5, 2):
            with self.subTest(system_dpi=dpi):
                self.run_desktop(check, system_dpi=lambda window: dpi)


if __name__ == "__main__":
    unittest.main()
