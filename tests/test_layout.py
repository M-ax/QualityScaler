"""Real Tk geometry regressions for resizable controls and status messages."""

import gc
import os
from queue import Queue
import sys
from threading import Event
import unittest
from unittest.mock import Mock, patch

from qualityscaler import gui
from qualityscaler.encoders import encoder_capabilities


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
                            gui.ROW_OUTPUT_FORMAT, gui.ROW_CODEC, gui.ROW_ENCODER_QUALITY):
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

    def test_encoder_controls_follow_codec_and_preserve_numeric_edits(self):
        def check(window):
            state = gui.app_state
            state.selected_video_crf.set("18.5")
            state.selected_video_bitrate.set("8500")
            for codec in (value for value in gui.video_codec_list if value != "----"):
                gui.App.select_video_codec_from_menu(codec)
                window.update()
                capabilities = encoder_capabilities(codec)
                preset = state.option_rows[gui.ROW_ENCODER_PRESET].winfo_children()[1]
                self.assertEqual(preset.cget("values"), list(capabilities.presets))
                slots = state.option_rows[gui.ROW_ENCODER_QUALITY].winfo_children()
                mode = slots[0].winfo_children()[1]
                self.assertEqual(mode.cget("values"), list(capabilities.rate_controls))
                for rate_control in capabilities.rate_controls:
                    gui.App.select_video_rate_control_from_menu(rate_control)
                    window.update()
                    slots = state.option_rows[gui.ROW_ENCODER_QUALITY].winfo_children()
                    entry = slots[1].winfo_children()[1]
                    self.assertEqual(entry.get(), "18.5" if rate_control == "CRF" else "8500")
                    options = gui._video_encoder_options()
                    self.assertEqual(options.rate_control, rate_control)
                # A rebuild must replace rather than accumulate controls.
                self.assertEqual(len(state.option_rows[gui.ROW_ENCODER_QUALITY].winfo_children()), 2)
            self.assertEqual(state.selected_video_crf.get(), "18.5")
            self.assertEqual(state.selected_video_bitrate.get(), "8500")
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
            self.assertEqual(bar.label.cget("state"), "disabled")
        self.run_desktop(check)


if __name__ == "__main__":
    unittest.main()
