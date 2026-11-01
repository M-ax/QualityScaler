"""Real Tk geometry regressions for resizable controls and status messages."""

import gc
import os
from queue import Queue
import sys
from threading import Event
from time import monotonic, sleep
import unittest
from unittest.mock import Mock, patch

from qualityscaler import gui
from qualityscaler.status import VideoProgress


@unittest.skipUnless(sys.platform == "win32" or os.environ.get("DISPLAY"),
                     "Layout tests need a desktop display")
class DesktopLayoutTests(unittest.TestCase):
    def run_desktop(self, check):
        manager = Mock()
        manager.Queue.return_value = Queue()
        manager.Event.return_value = Event()

        def smoke(window):
            try:
                check(window)
            finally:
                # CTk's library-wide appearance/DPI timers otherwise outlive
                # this test interpreter when another desktop is constructed.
                for handle in window.tk.splitlist(window.tk.call("after", "info")):
                    # Leave command deletion to the owning widget's destroy;
                    # root.after_cancel would delete child-owned Tcl commands.
                    window.tk.call("after", "cancel", handle)

        try:
            with patch.object(gui, "multiprocessing_Manager", return_value=manager), \
                    patch.object(gui, "load_user_preferences", return_value=gui.UserPreferences()), \
                    patch.object(gui.GPU, "detect"), \
                    patch.object(gui.GPU, "menu_list", return_value=["GeForce RTX 5080"]), \
                    patch.object(gui.GPU, "default", return_value="GeForce RTX 5080"), \
                    patch.object(gui.GPU, "vram_for", return_value=16), \
                    patch.object(gui.GPU, "codec_for", return_value="h264_nvenc"), \
                    patch.object(gui, "get_AI_engine_info", return_value="QualityScaler layout test"), \
                    patch.object(gui, "allow_sleep"), \
                    patch("customtkinter.CTk.mainloop", smoke):
                gui.main()
        finally:
            gui.apply_app_zoom(1.0)
            # Release every Tk variable/image on its owning thread, not during
            # a later worker's unrelated garbage collection.
            gui.app_state = None
            gc.collect()
        manager.shutdown.assert_called_once()

    def test_paired_controls_do_not_overlap_at_any_app_zoom(self):
        def check(window):
            for zoom in gui.zoom_option_list:
                gui.App.select_app_zoom(zoom)
                window.update()
                for row in (gui.ROW_RESOLUTION, gui.ROW_GPU,
                            gui.ROW_OUTPUT_FORMAT, gui.ROW_CODEC):
                    background = gui.app_state.option_rows[row]
                    slots = [child for child in background.winfo_children()
                             if isinstance(child, gui.CTkFrame)]
                    self.assertEqual(len(slots), 2)
                    previous_right = None
                    for slot in slots:
                        info, control = slot.winfo_children()
                        with self.subTest(zoom=zoom, row=row, column=slot.grid_info()["column"]):
                            self.assertLessEqual(info.winfo_x() + info.winfo_width(),
                                                 control.winfo_x())
                            self.assertLessEqual(control.winfo_x() + control.winfo_width(),
                                                 slot.winfo_width())
                            self.assertLessEqual(slot.winfo_x() + slot.winfo_width(),
                                                 background.winfo_width())
                            if previous_right is not None:
                                self.assertLessEqual(previous_right, slot.winfo_x())
                            previous_right = slot.winfo_x() + slot.winfo_width()
                            if isinstance(control, gui.CTkOptionMenu):
                                self.assertFalse(control.cget("dynamic_resizing"))
                                self.assertTrue(control.get())
                                self.assertGreater(control._text_label.winfo_width(), 20)
        self.run_desktop(check)

    def test_resize_keeps_status_and_action_visible_and_settings_scrollable(self):
        def check(window):
            self.assertEqual(tuple(bool(value) for value in window.resizable()), (True, True))
            previous = None
            for width, height in ((1000, 540), (1400, 900), (1000, 675)):
                window.geometry(f"{width}x{height}")
                window.update()
                state = gui.app_state
                bar = state.status_bar
                self.assertEqual(bar.winfo_width(), window.winfo_width())
                self.assertEqual(bar.winfo_y() + bar.winfo_height(), window.winfo_height())
                input_bottom = state.input_panel.winfo_rooty() + state.input_panel.winfo_height()
                self.assertLessEqual(input_bottom, bar.winfo_rooty())
                action_bottom = state.action_button.winfo_rooty() + state.action_button.winfo_height()
                self.assertLessEqual(action_bottom, bar.winfo_rooty())
                self.assertGreater(state.action_button.winfo_width(), 100)
                self.assertFalse(gui.accept_input_drop(type("Drop", (), {
                    "x_root": bar.winfo_rootx() + 10,
                    "y_root": bar.winfo_rooty() + 1,
                })()))
                if previous is not None and width > previous[0]:
                    self.assertGreater(state.input_panel.winfo_width(), previous[1])
                previous = (width, state.input_panel.winfo_width())
                canvas = state.settings_content._parent_canvas
                if height == 540:
                    self.assertLess(canvas.yview()[1] - canvas.yview()[0], 1)
                    canvas.yview_moveto(1)
                    window.update()
                    output_row = state.option_rows[gui.ROW_OUTPUT_PATH]
                    self.assertLessEqual(
                        output_row.winfo_rooty() + output_row.winfo_height(),
                        canvas.winfo_rooty() + canvas.winfo_height(),
                    )
        self.run_desktop(check)

    def test_long_status_wraps_and_shrinks_without_truncating_text(self):
        message = "1. Upscaling video — " + ("long filename and processing details " * 25)

        def check(window):
            window.geometry("1000x675")
            state = gui.app_state
            state.info_message.set(message)
            window.update()
            label = state.status_bar.label
            self.assertEqual(label.get("1.0", "end-1c"), message)
            long_height = state.status_bar.winfo_height()
            self.assertGreater(long_height, 36)
            self.assertLessEqual(label.winfo_x() + label.winfo_width(),
                                 state.status_bar.winfo_width())
            state.info_message.set("Ready")
            window.update()
            self.assertLess(state.status_bar.winfo_height(), long_height)
            self.assertEqual(state.status_bar.winfo_y() + state.status_bar.winfo_height(),
                             window.winfo_height())
        self.run_desktop(check)

    def test_initial_short_status_does_not_reserve_extra_lines(self):
        def check(window):
            window.update()
            bar = gui.app_state.status_bar
            self.assertEqual(bar.label.get("1.0", "end-1c"), "Hi :)")
            self.assertEqual(bar.label.cget("height"), 24)
            self.assertLessEqual(bar.winfo_height(), 40)
            self.assertFalse(bar.video_rows.winfo_ismapped())
        self.run_desktop(check)

    def test_cache_conflict_dialog_shows_explicit_clear_and_cancel_actions(self):
        def check(window):
            for use_failed_settings in (False, True):
                with self.subTest(use_failed_settings=use_failed_settings):
                    self._check_cache_conflict_dialog(window, use_failed_settings)
        self.run_desktop(check)

    def _check_cache_conflict_dialog(self, window, use_failed_settings):
        command = Mock()
        gui.show_error_message(
            "Video: C:/videos/a long video name.mp4\n\n"
            "The cached job has different AI, resize, tile, blending, or sharpening settings.",
            clear_cache_command=command,
        )
        dialog = next(child for child in window.winfo_children()
                      if isinstance(child, gui.MessageBox))
        try:
            # Run delayed construction deterministically on the Tk thread.
            if dialog._create_after is not None:
                dialog.after_cancel(dialog._create_after)
                dialog._create_widgets()
            window.update()
            choice = next(child for child in dialog.winfo_children()
                          if isinstance(child, gui.CTkCheckBox))
            self.assertEqual(choice.cget("text"), "Use settings from failed run")
            self.assertFalse(choice.get(), "Reusing failed-run settings must be opt-in")
            buttons = {
                child.cget("text"): child
                for frame in dialog.winfo_children() if isinstance(frame, gui.CTkFrame)
                for child in frame.winfo_children() if isinstance(child, gui.CTkButton)
            }
            self.assertEqual(set(buttons), {"Copy log", "Clear cache and continue", "Cancel"})
            # CTk briefly withdraws Windows dialogs while applying the
            # titlebar theme; let its timer finish before checking visibility.
            deadline = monotonic() + 5
            while not all(button.winfo_ismapped() for button in buttons.values()):
                self.assertLess(monotonic(), deadline, "The error dialog did not become visible")
                window.update()
                sleep(0.001)
            command.assert_not_called()
            for button in buttons.values():
                self.assertTrue(button.winfo_ismapped())
                self.assertLessEqual(
                    button.winfo_rootx() + button.winfo_width(),
                    dialog.winfo_rootx() + dialog.winfo_width(),
                )
            self.assertTrue(choice.winfo_ismapped())
            if use_failed_settings:
                choice.select()
            buttons["Clear cache and continue"].invoke()
            command.assert_called_once_with(use_failed_settings)
            self.assertFalse(dialog.winfo_exists())
        finally:
            if dialog.winfo_exists():
                dialog.destroy()

    def test_concurrent_video_rows_fit_at_all_zooms_without_hiding_stop(self):
        message = "2. Upscaling video — " + ("long processing details " * 100)

        def check(window):
            state = gui.app_state
            gui.App.place_stop_button()
            for zoom in gui.zoom_option_list:
                gui.App.select_app_zoom(zoom)
                window.geometry("1000x540")
                gui._display_status(VideoProgress(message, 120, 35, 100, False))
                window.update()
                bar = state.status_bar
                with self.subTest(zoom=zoom):
                    self.assertEqual(bar.label.get("1.0", "end-1c"), message)
                    self.assertTrue(bar.video_rows.winfo_ismapped())
                    self.assertEqual(bar.extraction_bar.get(), 1)
                    self.assertEqual(bar.upscaling_bar.get(), 0.35)
                    self.assertIn("120 / ~100 frames", bar.extraction_label.cget("text"))
                    self.assertIn("35 / ~100 frames", bar.upscaling_label.cget("text"))
                    self.assertEqual(bar.extraction_bar.winfo_x(), bar.upscaling_bar.winfo_x())
                    for label, progress in (
                        (bar.extraction_label, bar.extraction_bar),
                        (bar.upscaling_label, bar.upscaling_bar),
                    ):
                        self.assertLessEqual(label.winfo_x() + label.winfo_width(),
                                             progress.winfo_x())
                        self.assertGreater(progress.winfo_width(), 50)
                        self.assertLessEqual(progress.winfo_x() + progress.winfo_width(),
                                             bar.video_rows.winfo_width())
                    self.assertLess(
                        bar.extraction_label.winfo_y() + bar.extraction_label.winfo_height(),
                        bar.upscaling_label.winfo_y() + bar.upscaling_label.winfo_height(),
                    )
                    self.assertLess(bar.winfo_height(), window.winfo_height() / 4)
                    self.assertEqual(bar.winfo_y() + bar.winfo_height(), window.winfo_height())
                    self.assertLessEqual(
                        state.action_button.winfo_rooty() + state.action_button.winfo_height(),
                        bar.winfo_rooty(),
                    )
                    bar.label.yview_moveto(1)
                    window.update()
                    self.assertEqual(bar.label._textbox.yview()[1], 1)
        self.run_desktop(check)

    def test_video_rows_remain_during_encoding_and_reset_for_images(self):
        def check(window):
            state = gui.app_state
            bar = state.status_bar
            for snapshot in (
                VideoProgress("1. Extracting video", 42, 17, None, False),
                VideoProgress("1. Upscaling video", 100, 90, 100, True),
                VideoProgress("1. Encoding upscaled video", 100, 100, 100, True),
            ):
                gui._display_status(snapshot)
                window.update()
                self.assertTrue(bar.video_rows.winfo_ismapped())
                self.assertEqual(bar.label.get("1.0", "end-1c"), snapshot.message)
            self.assertTrue(bar.extraction_label.cget("text").endswith("complete"))
            self.assertTrue(bar.upscaling_label.cget("text").endswith("complete"))
            self.assertEqual(bar.extraction_bar.get(), 1)
            self.assertEqual(bar.upscaling_bar.get(), 1)
            video_height = bar.winfo_height()
            gui._display_status("2. Upscaling image")
            window.update()
            self.assertFalse(bar.video_rows.winfo_ismapped())
            self.assertLess(bar.winfo_height(), video_height)
            self.assertLessEqual(bar.winfo_height(), 40)
            self.assertEqual(bar.extraction_bar.get(), 0)
            self.assertEqual(bar.upscaling_bar.get(), 0)
        self.run_desktop(check)

    def test_extreme_status_remains_scrollable_without_hiding_stop(self):
        message = "long filename and processing details " * 250

        def check(window):
            window.geometry("1000x540")
            state = gui.app_state
            gui.App.place_stop_button()
            state.info_message.set(message)
            window.update()
            bar = state.status_bar
            self.assertEqual(bar.label.get("1.0", "end-1c"), message)
            self.assertLess(bar.winfo_height(), window.winfo_height() / 4)
            self.assertEqual(bar.winfo_y() + bar.winfo_height(), window.winfo_height())
            self.assertLessEqual(
                state.action_button.winfo_rooty() + state.action_button.winfo_height(),
                bar.winfo_rooty(),
            )
            bar.label.yview_moveto(1)
            window.update()
            self.assertEqual(bar.label._textbox.yview()[1], 1)
            self.assertEqual(bar.label._textbox.cget("state"), "disabled")
        self.run_desktop(check)


if __name__ == "__main__":
    unittest.main()
