"""Desktop contracts without mouse automation or inference/GPU execution."""

from dataclasses import fields
from inspect import signature
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qualityscaler import gui
from qualityscaler.catalog import MODEL_NAMES, VERSION, VRAM_model_usage
from qualityscaler.config import ProcessingConfig
from qualityscaler.jobs import upscale_orchestrator
from qualityscaler.status import CacheSettingsConflict, VideoProgress
from qualityscaler.widgets import StatusBar


def variable(value):
    result = Mock()
    result.get.return_value = value
    return result


def make_state():
    state = gui.AppState(
        preferences=gui.UserPreferences(),
        window=Mock(),
        info_message=Mock(),
        selected_output_path=variable(gui.OUTPUT_PATH_CODED),
        selected_input_resize_factor=variable("50"),
        selected_output_resize_factor=variable("100"),
        selected_VRAM_limiter=variable("4"),
        selected_video_codec=variable("x264"),
        process_status_q=Queue(),
        event_stop_upscale_process=Event(),
    )
    state.window.tk.splitlist.return_value = ()
    return state


class StateTestCase(unittest.TestCase):
    def setUp(self):
        self.state = make_state()
        patcher = patch.object(gui, "app_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)


class DesktopBackendWiringTests(StateTestCase):
    def test_spawn_bootstrap_does_not_import_gui_or_inference_runtime(self):
        launcher = Path(__file__).resolve().parents[1] / "QualityScaler.py"
        script = """
import builtins, runpy, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'customtkinter', 'tkinter', 'onnxruntime'}:
        raise AssertionError('Spawn imported GUI/runtime: ' + name)
    if name == 'qualityscaler.gui':
        raise AssertionError('Spawn imported desktop module')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
runpy.run_path(sys.argv[1], run_name='__mp_main__')
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(launcher)],
            capture_output=True, text=True, check=False, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_config_and_catalog_import_without_gui_or_runtime(self):
        script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'customtkinter', 'tkinter', 'onnxruntime', 'cv2', 'numpy'}:
        raise AssertionError('Pure config imported backend: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from qualityscaler.config import ProcessingConfig
from qualityscaler.catalog import MODEL_NAMES, VERSION
assert VERSION == '2026.4'
assert len(MODEL_NAMES) == 9
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            check=False, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_start_uses_modular_orchestrator_with_exact_positional_contract(self):
        self.state.status_bar = Mock()
        process = Mock()
        config = ProcessingConfig(
            ["clip.mp4"], "output directory", "LVAx2", 2, 0.5, 1.0,
            "0", 800, 0.3, True, ".png", ".mp4", "h264_nvenc",
        )
        with patch.object(gui, "build_processing_config", return_value=config), \
                patch.object(gui.App, "place_stop_button"), \
                patch.object(gui, "Process", return_value=process) as constructor, \
                patch.object(gui, "prevent_sleep") as prevent, \
                patch.object(gui, "_schedule"):
            gui.upscale_button_command()
        self.assertIs(constructor.call_args.kwargs["target"], upscale_orchestrator)
        args = constructor.call_args.kwargs["args"]
        self.assertEqual(args, (
            self.state.process_status_q, self.state.event_stop_upscale_process,
            ["clip.mp4"], "output directory", "LVAx2", 2, 0.5, 1.0,
            "0", 800, 0.3, True, ".png", ".mp4", "h264_nvenc",
        ))
        bound = signature(upscale_orchestrator).bind(*args)
        self.assertIs(bound.arguments["event_stop_upscale_process"],
                      self.state.event_stop_upscale_process)
        self.assertEqual(bound.arguments["selected_file_list"], ["clip.mp4"])
        self.assertEqual(bound.arguments["selected_sharpening_amount"], 0.3)
        self.assertNotIn("video_frames_and_info_q", bound.arguments)
        self.assertNotIn("video_frames_and_info_q", vars(self.state))
        prevent.assert_called_once()
        process.start.assert_called_once()
        self.assertTrue(self.state.processing)
        self.state.status_bar.set_video_progress.assert_called_once_with(None)
        self.assertIs(self.state.processing_config, config)
        self.assertNotIn("kwargs", constructor.call_args.kwargs)

    def test_desktop_launcher_actually_spawns_gui_free_workers(self):
        root = Path(__file__).resolve().parents[1]
        script = """
import os, runpy, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'tests'))
from spawn_worker_probe import process_imports
def launch_workers():
    from qualityscaler.video import create_executor
    with create_executor(max_workers=1) as executor:
        result = executor.submit(process_imports).result(timeout=15)
    assert result['pid'] != os.getpid(), result
    assert not result['gui_modules'], result
with patch.dict(sys.modules, {'qualityscaler.gui': SimpleNamespace(main=launch_workers)}):
    runpy.run_path(str(root / 'QualityScaler.py'), run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(root)], capture_output=True,
            text=True, check=False, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stop_waits_for_checkpoint_before_hard_kill(self):
        process = Mock(pid=123)
        process.is_alive.return_value = False
        self.state.process_upscale_orchestrator = process
        self.state.process_status_q.put("old progress")
        with patch.object(gui, "allow_sleep") as allow:
            gui.stop_upscale_process()
        process.join.assert_called_once_with(timeout=10)
        process.kill.assert_not_called()
        self.assertTrue(self.state.process_status_q.empty())
        self.assertFalse(self.state.event_stop_upscale_process.is_set())
        self.assertIsNone(self.state.process_upscale_orchestrator)
        allow.assert_called_once()

    def test_hung_worker_fallback_kills_ffmpeg_descendants_and_joins(self):
        process = Mock(pid=123)
        process.is_alive.return_value = True
        self.state.process_upscale_orchestrator = process
        child = Mock()
        with patch.object(gui, "allow_sleep"), \
                patch.object(gui, "psutil_Process") as lookup:
            lookup.return_value.children.return_value = [child]
            gui.stop_upscale_process()
        lookup.return_value.children.assert_called_once_with(recursive=True)
        child.kill.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.join.call_count, 2)
        self.assertFalse(self.state.event_stop_upscale_process.is_set())

    def test_progress_highlights_card_on_tk_poll(self):
        self.state.processing = True
        self.state.process_upscale_orchestrator = Mock()
        self.state.file_widget = Mock()
        self.state.status_bar = Mock()
        self.state.process_status_q.put("2. Upscaling video 30%")
        with patch.object(gui, "_schedule") as schedule:
            gui.check_upscale_steps()
        self.state.info_message.set.assert_called_with("2. Upscaling video 30%")
        self.state.file_widget.highlight_active_file.assert_called_once_with(2)
        self.state.status_bar.set_video_progress.assert_called_once_with(None)
        schedule.assert_called_once_with(250, gui.check_upscale_steps)

    def test_video_snapshot_updates_both_rows_and_highlight_from_message(self):
        self.state.processing = True
        self.state.process_upscale_orchestrator = Mock()
        self.state.file_widget = Mock()
        self.state.status_bar = Mock()
        progress = VideoProgress("2. Upscaling video", 90, 30, 100, False)
        self.state.process_status_q.put(progress)
        with patch.object(gui, "_schedule") as schedule:
            gui.check_upscale_steps()
        self.state.info_message.set.assert_called_once_with(progress.message)
        self.state.status_bar.set_video_progress.assert_called_once_with(progress)
        self.state.file_widget.highlight_active_file.assert_called_once_with(2)
        schedule.assert_called_once_with(250, gui.check_upscale_steps)

    def test_encoding_snapshot_stays_structured_and_image_progress_resets_rows(self):
        self.state.processing = True
        self.state.process_upscale_orchestrator = Mock()
        self.state.file_widget = Mock()
        self.state.status_bar = Mock()
        progress = VideoProgress("1. Encoding upscaled video", 100, 100, 100, True)
        with patch.object(gui, "_schedule"):
            self.state.process_status_q.put(progress)
            gui.check_upscale_steps()
            self.state.status_bar.set_video_progress.assert_called_with(progress)
            self.state.process_status_q.put("2. Upscaling image")
            gui.check_upscale_steps()
        self.state.info_message.set.assert_called_with("2. Upscaling image")
        self.state.status_bar.set_video_progress.assert_called_with(None)
        self.state.file_widget.highlight_active_file.assert_called_with(2)

    def test_progress_after_job_settles_cannot_repopulate_rows(self):
        self.state.status_bar = Mock()
        self.state.process_status_q.put(VideoProgress("1. Upscaling video", 10, 2, None, False))
        gui.check_upscale_steps()
        self.state.status_bar.set_video_progress.assert_not_called()
        self.state.info_message.set.assert_not_called()

    def test_completed_status_restores_sleep_and_refreshes_resume_badges(self):
        self.state.processing = True
        self.state.process_upscale_orchestrator = Mock(pid=123, exitcode=0)
        self.state.process_upscale_orchestrator.is_alive.return_value = False
        self.state.process_status_q.put(gui.COMPLETED_STATUS)
        self.state.file_widget = Mock()
        with patch.object(gui, "allow_sleep") as allow, \
                patch.object(gui, "Thread") as notify, \
                patch.object(gui, "update_file_widget") as refresh, \
                patch.object(gui.App, "place_upscale_button"):
            gui.check_upscale_steps()
        self.state.info_message.set.assert_called_with("All files completed! :)")
        self.state.file_widget.clear_active_highlight.assert_called_once()
        notify.assert_called_once_with(
            target=gui.show_completion_notification,
            name="QualityScaler-completion-notification", daemon=True,
        )
        notify.return_value.start.assert_called_once()
        allow.assert_called_once()
        refresh.assert_called_once()
        self.assertFalse(self.state.processing)

    def test_start_failure_restores_idle_ui_and_sleep(self):
        config = ProcessingConfig(
            ["image.png"], gui.OUTPUT_PATH_CODED, "LVAx2", 1, 0.5, 1.0,
            "0", 800, 0.3, True, ".png", ".mp4", "x264",
        )
        process = Mock(pid=None)
        process.start.side_effect = RuntimeError("spawn failed")
        with patch.object(gui, "build_processing_config", return_value=config), \
                patch.object(gui, "Process", return_value=process), \
                patch.object(gui, "prevent_sleep"), patch.object(gui, "allow_sleep") as allow, \
                patch.object(gui, "show_error_message") as error, \
                patch.object(gui.App, "place_stop_button"), \
                patch.object(gui.App, "place_upscale_button") as idle:
            gui.upscale_button_command()
        error.assert_called_once_with("spawn failed")
        idle.assert_called_once()
        allow.assert_called_once()
        self.assertFalse(gui.inputs_are_busy())

    def test_busy_start_cannot_spawn_second_orchestrator(self):
        self.state.processing = True
        with patch.object(gui, "Process") as constructor:
            gui.upscale_button_command()
        constructor.assert_not_called()


class CacheConflictDialogTests(StateTestCase):
    def test_clear_action_explains_scope_and_cancel_preserves_cache(self):
        command = Mock()
        with patch.object(gui, "MessageBox") as dialog:
            gui.show_error_message("Cached settings differ", clear_cache_command=command)
        self.assertEqual(dialog.call_args.kwargs, {
            "action_text": "Clear cache and continue", "action_command": command,
            "action_option_text": "Use settings from failed run",
        })
        self.assertIn("Current UI settings are used by default", dialog.call_args.args[2])
        details = "\n".join(dialog.call_args.args[4])
        self.assertIn("cached source and AI frames", details)
        self.assertIn("Other caches are untouched", details)
        self.assertIn("Any existing output is replaced only after the new video encodes successfully", details)
        self.assertIn("Cancel leaves the cache unchanged", details)
        self.assertIn("old cache is kept", details)
        command.assert_not_called()

    def test_generic_error_has_no_cache_deletion_action(self):
        with patch.object(gui, "MessageBox") as dialog:
            gui.show_error_message("The cached job has different settings")
        self.assertEqual(dialog.call_args.kwargs, {})

    def test_action_dismisses_dialog_before_starting_once(self):
        events = []
        dialog = object.__new__(gui.MessageBox)
        dialog.grab_release = lambda: events.append("release")
        dialog.destroy = lambda: events.append("destroy")
        dialog._action_command = lambda: events.append("retry")
        dialog._action_event()
        dialog._action_event()
        self.assertEqual(events, ["release", "destroy", "retry"])

    def test_settings_choice_is_read_before_destroy_and_sent_once(self):
        for use_failed_settings in (False, True):
            with self.subTest(use_failed_settings=use_failed_settings):
                command = Mock()
                option = Mock()
                option.get.return_value = use_failed_settings
                dialog = object.__new__(gui.MessageBox)
                dialog._action_command = command
                dialog._action_option = option
                dialog.grab_release = Mock()
                dialog.destroy = lambda: setattr(
                    option.get, "side_effect", AssertionError("Read choice after destroying dialog"),
                )
                dialog._action_event()
                dialog._action_event()
                command.assert_called_once_with(use_failed_settings)
                option.get.assert_called_once()

    def test_cancel_and_window_close_disarm_the_action(self):
        for dismiss in ("_ok_event", "_on_closing"):
            with self.subTest(dismiss=dismiss):
                command = Mock()
                dialog = object.__new__(gui.MessageBox)
                dialog.grab_release = Mock()
                dialog.destroy = Mock()
                dialog._action_command = command
                getattr(dialog, dismiss)()
                dialog._action_event()
                command.assert_not_called()
                dialog.destroy.assert_called_once()

    def test_retry_uses_current_ui_settings_by_default_and_preserves_batch_numbering(self):
        config = ProcessingConfig(
            ["first.png", "second.mp4", "third.mp4"], gui.OUTPUT_PATH_CODED, "LVAx2",
            2, 0.5, 1.0, "0", 800, 0.3, True, ".png", ".mp4", "x264",
        )
        self.state.preferences.ai_model = "BSRGANx2"
        self.state.selected_input_resize_factor.get.return_value = "75"
        self.state.selected_output_resize_factor.get.return_value = "125"
        self.state.file_widget = Mock()
        self.state.file_widget.get_selected_file_list.return_value = ["unrelated.mp4"]
        with patch.object(gui, "build_processing_config") as build, \
                patch.object(gui, "Process", return_value=Mock()) as constructor, \
                patch.object(gui, "prevent_sleep"), \
                patch.object(gui.App, "place_stop_button"), \
                patch.object(gui, "_schedule"):
            gui._retry_after_cache_conflict(config, 2)
        build.assert_not_called()
        self.state.file_widget.get_selected_file_list.assert_not_called()
        self.assertEqual(constructor.call_args.kwargs["kwargs"], {
            "retry_from": 2, "clear_cache": True,
        })
        args = constructor.call_args.kwargs["args"]
        self.assertEqual(args[2], ["first.png", "second.mp4", "third.mp4"])
        self.assertEqual(args[4], "BSRGANx2")
        self.assertEqual(args[6:8], (0.75, 1.25))
        self.assertIsNot(self.state.processing_config, config)

    def test_failed_run_settings_require_explicit_opt_in(self):
        config = ProcessingConfig(
            ["first.png", "second.mp4", "third.mp4"], "original output", "LVAx2",
            2, 0.5, 1.0, "0", 800, 0.3, True, ".png", ".mp4", "x264",
        )
        self.state.preferences.ai_model = "Changed after failure"
        with patch.object(gui, "_processing_config") as build, \
                patch.object(gui, "Process", return_value=Mock()) as constructor, \
                patch.object(gui, "prevent_sleep"), \
                patch.object(gui.App, "place_stop_button"), \
                patch.object(gui, "_schedule"):
            gui._retry_after_cache_conflict(config, 2, use_failed_settings=True)
        build.assert_not_called()
        self.assertEqual(constructor.call_args.kwargs["kwargs"], {
            "retry_from": 2, "clear_cache": True,
        })
        args = constructor.call_args.kwargs["args"]
        self.assertEqual(args[2], ["first.png", "second.mp4", "third.mp4"])
        self.assertEqual(args[3:5], ("original output", "LVAx2"))
        self.assertIs(self.state.processing_config, config)

    def test_current_output_folder_change_does_not_authorize_clearing_another_cache(self):
        config = ProcessingConfig(
            ["first.png", "second.mp4"], "failed output", "LVAx2",
            2, 0.5, 1.0, "0", 800, 0.3, True, ".png", ".mp4", "x264",
        )
        self.state.selected_output_path.get.return_value = "new output"
        with patch.object(gui, "_start_upscale") as start, \
                patch.object(gui, "check_disk_space", return_value=None) as disk:
            gui._retry_after_cache_conflict(config, 2)
        args, kwargs = start.call_args
        self.assertEqual(args[0].selected_output_path, "new output")
        self.assertEqual(kwargs, {"retry_from": 2, "clear_cache": False})
        disk.assert_called_once_with("new output", ["second.mp4"])

    def test_equivalent_output_locations_allow_clearing_the_reported_cache(self):
        with TemporaryDirectory() as directory:
            video = str(Path(directory) / "second.mp4")
            config = ProcessingConfig(
                ["first.png", video], gui.OUTPUT_PATH_CODED, "LVAx2",
                2, 0.5, 1.0, "0", 800, 0.3, True, ".png", ".mp4", "x264",
            )
            self.state.selected_output_path.get.return_value = directory
            with patch.object(gui, "_start_upscale") as start, \
                    patch.object(gui, "check_disk_space", return_value=None):
                gui._retry_after_cache_conflict(config, 2)
            self.assertEqual(start.call_args.kwargs, {"retry_from": 2, "clear_cache": True})

    def test_invalid_current_settings_do_not_clear_or_restart(self):
        self.state.preferences.ai_model = "Invalid model"
        with patch.object(gui, "_start_upscale") as start:
            gui._retry_after_cache_conflict(Mock(selected_file_list=["clip.mp4"]), 1)
        start.assert_not_called()
        self.state.info_message.set.assert_called_once_with("Please select the AI model")

    def test_low_disk_space_blocks_retry_without_clearing(self):
        with patch.object(gui, "_start_upscale") as start, \
                patch.object(gui, "check_disk_space", return_value="Free up space"), \
                patch.object(gui, "show_disk_space_error_message") as error:
            gui._retry_after_cache_conflict(Mock(selected_file_list=["clip.mp4"]), 1)
        start.assert_not_called()
        error.assert_called_once_with("Free up space")

    def test_retry_cannot_start_during_another_job_or_shutdown(self):
        for flag in ("processing", "closing"):
            with self.subTest(flag=flag):
                setattr(self.state, flag, True)
                with patch.object(gui, "_start_upscale") as start:
                    gui._retry_after_cache_conflict(Mock(), 2)
                start.assert_not_called()
                setattr(self.state, flag, False)


class InputSelectionTests(StateTestCase):
    def setUp(self):
        super().setUp()
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.first = self.root / "résumé {one}.PNG"
        self.second = self.root / "second clip.MP4"
        self.first.write_bytes(b"preview is mocked")
        self.second.write_bytes(b"preview is mocked")
        self.views = []
        self.reject = set()
        self.status = self.state.info_message
        self.placeholder = (Mock(), Mock(), Mock())
        self.state.input_widgets = self.placeholder
        self.state.input_panel = Mock()
        self.state.input_panel.winfo_rootx.return_value = 100
        self.state.input_panel.winfo_rooty.return_value = 200
        self.state.input_panel.winfo_width.return_value = 500
        self.state.input_panel.winfo_height.return_value = 639
        self.state.window.winfo_rootx.return_value = 100
        self.state.window.winfo_rooty.return_value = 200
        self.state.window.winfo_width.return_value = 1000
        self.state.window.winfo_height.return_value = 675
        patcher = patch.object(gui, "get_values_for_file_widget", return_value=(4, 50, 100))
        patcher.start()
        self.addCleanup(patcher.stop)

        def make_view(**kwargs):
            view = Mock()
            paths = kwargs["selected_file_list"]
            view.unreadable_files = [path for path in paths if path in self.reject]
            view.get_selected_file_list.return_value = [
                path for path in paths if path not in self.reject
            ]
            self.views.append(view)
            return view

        patcher = patch.object(gui, "FileWidget", side_effect=make_view)
        self.constructor = patcher.start()
        self.addCleanup(patcher.stop)

    def test_picker_and_repeated_drops_share_append_and_duplicate_rules(self):
        with patch.object(gui.filedialog, "askopenfilenames", return_value=(str(self.first),)):
            gui.open_files_action()
        first_view = self.views[0]
        self.assertTrue(gui.select_input_files((str(self.first), str(self.second))))
        self.assertEqual(self.state.selected_file_list,
                         [str(self.first.resolve()), str(self.second.resolve())])
        first_view.destroy.assert_called_once()
        for widget in self.placeholder:
            widget.destroy.assert_called_once()
        self.assertTrue(gui.select_input_files((str(self.first),)))
        self.assertEqual(self.constructor.call_count, 2)

    def test_picker_cancellation_leaves_selection_and_status_untouched(self):
        gui.select_input_files([self.first])
        view = self.state.file_widget
        self.status.reset_mock()
        with patch.object(gui.filedialog, "askopenfilenames", return_value=()):
            gui.open_files_action()
        self.assertIs(self.state.file_widget, view)
        view.destroy.assert_not_called()
        self.status.set.assert_not_called()

    def test_invalid_drop_leaves_existing_selection_untouched(self):
        gui.select_input_files([self.first])
        view = self.state.file_widget
        notes = self.root / "notes.txt"
        notes.write_text("ignore")
        self.assertFalse(gui.select_input_files([notes, self.root, "missing.jpg"]))
        self.assertIs(self.state.file_widget, view)
        self.assertEqual(self.state.selected_file_list, [str(self.first.resolve())])
        view.destroy.assert_not_called()

    def test_unreadable_new_media_does_not_replace_a_valid_selection(self):
        gui.select_input_files([self.first])
        view = self.state.file_widget
        self.reject.add(str(self.second.resolve()))
        self.assertFalse(gui.select_input_files([self.second]))
        self.assertIs(self.state.file_widget, view)
        view.destroy.assert_not_called()
        self.views[-1].destroy.assert_called_once()

    def test_active_job_refuses_picker_drop_and_clear(self):
        gui.select_input_files([self.first])
        view = self.state.file_widget
        self.state.process_upscale_orchestrator = SimpleNamespace(is_alive=lambda: True)
        with patch.object(gui.filedialog, "askopenfilenames") as picker:
            gui.open_files_action()
        picker.assert_not_called()
        self.assertFalse(gui.select_input_files([self.second]))
        self.assertFalse(gui.clear_input_files())
        self.assertIs(self.state.file_widget, view)
        view.destroy.assert_not_called()

    def test_starting_and_stopping_are_busy_even_without_live_process(self):
        for name in ("processing", "stopping"):
            with self.subTest(state=name):
                setattr(self.state, name, True)
                self.assertFalse(gui.select_input_files([self.first]))
                self.assertFalse(gui.clear_input_files())
                setattr(self.state, name, False)

    def test_drop_only_accepts_the_left_input_pane(self):
        self.assertTrue(gui.accept_input_drop(SimpleNamespace(x_root=350, y_root=300)))
        for x, y in ((600, 300), (850, 300), (99, 300), (350, 199),
                     (350, 839), (350, 875)):
            with self.subTest(x=x, y=y):
                self.assertFalse(gui.accept_input_drop(SimpleNamespace(x_root=x, y_root=y)))

    def test_clear_resets_widget_and_queue_before_showing_placeholder(self):
        gui.select_input_files([self.first])
        view = self.state.file_widget
        self.state.status_bar = Mock()
        with patch.object(gui.App, "place_loadFile_section") as placeholder:
            self.assertTrue(gui.clear_input_files())
        self.assertIsNone(self.state.file_widget)
        self.assertEqual(self.state.selected_file_list, [])
        view.destroy.assert_called_once()
        placeholder.assert_called_once()
        self.state.status_bar.set_video_progress.assert_called_once_with(None)

    def test_new_selection_clears_video_rows_from_previous_job(self):
        self.state.status_bar = Mock()
        self.assertTrue(gui.select_input_files([self.first]))
        self.state.status_bar.set_video_progress.assert_called_once_with(None)


class VideoStatusWidgetTests(unittest.TestCase):
    """Snapshot rendering rules without a Tk interpreter or display."""

    def setUp(self):
        self.bar = object.__new__(StatusBar)
        self.bar.video_rows = Mock()
        self.bar.extraction_label = Mock()
        self.bar.upscaling_label = Mock()
        self.bar.extraction_bar = Mock()
        self.bar.upscaling_bar = Mock()
        self.bar._queue_fit = Mock()
        self.bar._video_progress = None

    def test_unknown_total_shows_counts_without_a_percentage_or_fake_fraction(self):
        self.bar.set_video_progress(VideoProgress("1. Video", 42, 17, None, False))
        self.bar.extraction_label.configure.assert_called_with(
            text="Extraction: 42 frames, total unknown — extracting"
        )
        self.bar.upscaling_label.configure.assert_called_with(
            text="AI upscaling: 17 frames, total unknown — upscaling"
        )
        self.bar.extraction_bar.set.assert_called_with(0)
        self.bar.upscaling_bar.set.assert_called_with(0)
        self.bar.video_rows.grid.assert_called_once_with()

    def test_estimated_total_is_marked_and_bars_clamp_without_losing_actual_counts(self):
        self.bar.set_video_progress(VideoProgress("1. Video", 120, 110, 100, False))
        self.bar.extraction_label.configure.assert_called_with(
            text="Extraction: 120 / ~100 frames — extracting"
        )
        self.bar.upscaling_label.configure.assert_called_with(
            text="AI upscaling: 110 / ~100 frames — upscaling"
        )
        self.bar.extraction_bar.set.assert_called_with(1)
        self.bar.upscaling_bar.set.assert_called_with(1)
        self.bar.set_video_progress(VideoProgress("1. Video", 130, 110, 130, True))
        self.bar.extraction_label.configure.assert_called_with(
            text="Extraction: 130 / 130 frames — complete"
        )
        self.bar.upscaling_label.configure.assert_called_with(
            text="AI upscaling: 110 / 130 frames — upscaling"
        )
        self.bar.upscaling_bar.set.assert_called_with(110 / 130)

    def test_waiting_for_frames_is_not_completion_when_estimate_is_reached(self):
        self.bar.set_video_progress(VideoProgress("1. Video", 100, 100, 100, False))
        self.bar.upscaling_label.configure.assert_called_with(
            text="AI upscaling: 100 / ~100 frames — waiting for frames"
        )
        self.bar.set_video_progress(VideoProgress("1. Video", 100, 100, 100, True))
        self.bar.upscaling_label.configure.assert_called_with(
            text="AI upscaling: 100 / 100 frames — complete"
        )

    def test_zero_estimate_does_not_divide_by_zero(self):
        self.bar.set_video_progress(VideoProgress("1. Video", 1, 0, 0, False))
        self.bar.extraction_bar.set.assert_called_with(0)
        self.bar.upscaling_bar.set.assert_called_with(0)

    def test_new_file_replaces_both_resume_counters_and_reset_clears_them(self):
        self.bar.set_video_progress(VideoProgress("1. Video", 100, 90, 100, True))
        self.bar.set_video_progress(VideoProgress("2. Video", 4, 2, 50, False))
        self.bar.extraction_bar.set.assert_called_with(4 / 50)
        self.bar.upscaling_bar.set.assert_called_with(2 / 50)
        self.bar.upscaling_label.configure.assert_called_with(
            text="AI upscaling: 2 / ~50 frames — upscaling"
        )
        self.bar.set_video_progress(None)
        self.bar.video_rows.grid_remove.assert_called_once_with()
        self.bar.extraction_label.configure.assert_called_with(text="")
        self.bar.upscaling_label.configure.assert_called_with(text="")
        self.bar.extraction_bar.set.assert_called_with(0)
        self.bar.upscaling_bar.set.assert_called_with(0)
        self.assertIsNone(self.bar._video_progress)

    def test_cleanup_preserves_counters_without_leaving_active_stage_wording(self):
        self.bar.set_video_progress(VideoProgress("1. Video", 100, 90, 100, True))
        for status in ("stopping", "finalizing", "stopped", "error", "job finished"):
            with self.subTest(status=status):
                self.bar.set_video_status(status)
                self.bar.extraction_label.configure.assert_called_with(
                    text="Extraction: 100 / 100 frames — complete"
                )
                self.bar.upscaling_label.configure.assert_called_with(
                    text=f"AI upscaling: 90 / 100 frames — {status}"
                )
                self.bar.upscaling_bar.set.assert_called_with(0.9)
        self.bar.set_video_progress(None)
        self.bar.set_video_status("error")
        self.bar.video_rows.grid_remove.assert_called_once_with()
        self.bar.upscaling_label.configure.assert_called_with(text="")


class PreferencesAndConfigTests(StateTestCase):
    def setUp(self):
        super().setUp()
        self.state.file_widget = Mock()
        self.state.file_widget.get_selected_file_list.return_value = ["clip.mp4"]
        patcher = patch.object(gui.GPU, "device_id_for", return_value="1")
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(gui, "check_disk_space", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_exact_upstream_config_fields_and_nine_model_catalog(self):
        self.assertEqual([field.name for field in fields(ProcessingConfig)], [
            "selected_file_list", "selected_output_path", "selected_AI_model",
            "selected_AI_multithreading", "input_resize_factor", "output_resize_factor",
            "selected_gpu", "tiles_resolution", "selected_sharpening_amount",
            "selected_keep_frames", "selected_image_extension",
            "selected_video_extension", "selected_video_codec",
        ])
        self.assertEqual(VERSION, "2026.4")
        self.assertEqual(MODEL_NAMES, (
            "LVAx2", "RealESR_Gx4", "RealESR_Ax4", "BSRGANx2", "BSRGANx4",
            "RealESRGANx4", "MSharpx4", "IRCNN_Mx1", "IRCNN_Lx1",
        ))
        self.assertEqual(set(MODEL_NAMES), set(VRAM_model_usage))

    def test_defaults_match_upstream_and_config_is_plain_values(self):
        preferences = self.state.preferences
        self.assertEqual(preferences.ai_model, "LVAx2")
        self.assertEqual(preferences.sharpening, "Low")
        self.assertTrue(preferences.keep_frames)
        config = gui.build_processing_config()
        self.assertEqual(config.selected_file_list, ["clip.mp4"])
        self.assertEqual(config.selected_gpu, "1")
        self.assertEqual(config.tiles_resolution, 800)
        self.assertEqual(config.input_resize_factor, 0.5)
        self.assertEqual(config.output_resize_factor, 1.0)
        self.assertEqual(config.selected_sharpening_amount, 0.3)
        self.assertEqual(config.selected_AI_multithreading, 1)

    def test_models_use_upstream_vram_multipliers(self):
        for model, multiplier in VRAM_model_usage.items():
            with self.subTest(model=model):
                self.state.preferences.ai_model = model
                self.assertEqual(gui.build_processing_config().tiles_resolution,
                                 int(multiplier * 4 * 100))

    def test_bad_numeric_input_is_reported_without_starting(self):
        for name in ("selected_input_resize_factor", "selected_output_resize_factor",
                     "selected_VRAM_limiter"):
            for value in ("bad", "0", "-1", "nan", "inf"):
                with self.subTest(variable=name, value=value):
                    original = getattr(self.state, name)
                    setattr(self.state, name, variable(value))
                    self.assertIsNone(gui.build_processing_config())
                    setattr(self.state, name, original)

    def test_resume_uses_current_video_settings_and_readonly_backend_api(self):
        with patch.object(gui, "get_resume_progress", return_value=42) as resume:
            self.assertEqual(gui.get_video_resume_progress("clip.mp4"), 42)
            path, output, settings = resume.call_args.args
            self.assertEqual(path, "clip.mp4")
            self.assertEqual(output, gui.OUTPUT_PATH_CODED)
            self.assertEqual(settings.model, "LVAx2")
            self.assertEqual(settings.gpu, "1")
            self.assertEqual(settings.sharpening_amount, 0.3)
            self.assertEqual(settings.tiles_resolution, 800)
            self.state.preferences.sharpening = "High"
            self.state.preferences.ai_multithreading = "4 threads"
            self.state.selected_output_path.get.return_value = "new output"
            gui.get_video_resume_progress("clip.mp4")
            self.assertEqual(resume.call_args.args[1], "new output")
            self.assertEqual(resume.call_args.args[2].sharpening_amount, 0.5)
            self.assertEqual(resume.call_args.args[2].workers, 4)

    def test_invalid_settings_suppress_resume_without_status_changes(self):
        self.state.selected_input_resize_factor.get.return_value = "not a number"
        with patch.object(gui, "get_resume_progress") as resume:
            self.assertIsNone(gui.get_video_resume_progress("clip.mp4"))
        resume.assert_not_called()
        self.state.info_message.set.assert_not_called()

    def test_gpu_selection_applies_detected_vram_and_vendor_codec(self):
        with patch.object(gui.GPU, "vram_for", return_value=12), \
                patch.object(gui.GPU, "codec_for", return_value="hevc_nvenc"), \
                patch.object(gui, "update_file_widget") as refresh:
            gui.App.select_gpu_from_menu("NVIDIA GPU")
        self.assertEqual(self.state.preferences.gpu, "NVIDIA GPU")
        self.assertEqual(self.state.preferences.vram_limiter, "12")
        self.state.selected_VRAM_limiter.set.assert_called_with("12")
        self.state.selected_video_codec.set.assert_called_with("hevc_nvenc")
        self.assertEqual(self.state.preferences.video_codec, "hevc_nvenc")
        refresh.assert_called_once()

    def test_no_gpu_detection_preserves_vram_fallback_and_software_codec(self):
        with patch.object(gui.GPU, "vram_for", return_value=None), \
                patch.object(gui.GPU, "codec_for", return_value=None):
            gui.apply_auto_vram_for_gpu("No GPU found", fallback="4")
            gui.apply_auto_codec_for_gpu("No GPU found")
        self.state.selected_VRAM_limiter.set.assert_called_with("4")
        self.assertEqual(self.state.preferences.video_codec, "x264")

    def test_zoom_updates_both_tk_scalings_and_preference(self):
        with patch.object(gui, "set_window_scaling") as windows, \
                patch.object(gui, "set_widget_scaling") as widgets:
            gui.App.select_app_zoom("125%")
        self.assertEqual(self.state.preferences.app_zoom, "125%")
        windows.assert_called_once_with(1.25)
        widgets.assert_called_once_with(1.25)

    def test_preferences_round_trip_upstream_keys_and_tolerate_corruption(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.json"
            with patch.object(gui, "USER_PREFERENCE_PATH", str(path)):
                self.state.preferences.app_zoom = "125%"
                self.state.preferences.sharpening = "High"
                gui.save_user_choices_in_json()
                data = json.loads(path.read_text())
                self.assertEqual(data["default_keep_frames"], "ON")
                self.assertNotIn("default_blending", data)
                loaded = gui.load_user_preferences()
                self.assertEqual(loaded.app_zoom, "125%")
                self.assertEqual(loaded.sharpening, "High")
                for contents in ("{broken", "[]", '{"default_app_zoom": "broken"}'):
                    path.write_text(contents)
                    self.assertEqual(gui.load_user_preferences().app_zoom, "100%")


class DiskSpaceTests(unittest.TestCase):
    def test_images_do_not_require_video_disk_space_budget(self):
        with patch.object(gui, "shutil_disk_usage") as usage:
            self.assertIsNone(gui.check_disk_space("output", ["image.PNG"]))
        usage.assert_not_called()

    def test_video_low_disk_blocks_start(self):
        with TemporaryDirectory() as directory:
            with patch.object(gui, "shutil_disk_usage",
                              return_value=SimpleNamespace(free=1024 ** 3)):
                warning = gui.check_disk_space(directory, ["clip.MP4"])
        self.assertIn("1.0 GB free", warning)

    def test_each_source_output_drive_is_checked(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            def disk(path):
                return SimpleNamespace(free=(3 if path == first else 1) * 1024 ** 3)
            with patch.object(gui, "shutil_disk_usage", side_effect=disk):
                warning = gui.check_disk_space(
                    gui.OUTPUT_PATH_CODED,
                    [str(Path(first) / "one.mp4"), str(Path(second) / "two.mp4")],
                )
        self.assertIsNotNone(warning)


class CardResourceTests(StateTestCase):
    def test_typing_coalesces_preview_rebuilds_until_last_edit(self):
        self.state.file_widget = Mock()
        with patch.object(gui, "_schedule", side_effect=["after#1", "after#2", "after#3"]) as schedule:
            for value in ("5", "50", "500"):
                self.state.selected_input_resize_factor.get.return_value = value
                gui.schedule_file_widget_update()
        self.state.file_widget.refresh_pipeline.assert_not_called()
        self.assertEqual(schedule.call_count, 3)
        self.assertEqual(self.state.preview_after_id, "after#3")
        self.state.window.after_cancel.assert_any_call("after#1")
        self.state.window.after_cancel.assert_any_call("after#2")
        callback = schedule.call_args.args[1]
        callback()
        self.state.file_widget.refresh_pipeline.assert_called_once()
        self.state.file_widget.set_input_resize_factor.assert_called_once_with(500)
        self.assertIsNone(self.state.preview_after_id)

    def test_discrete_change_cancels_pending_typing_refresh(self):
        self.state.file_widget = Mock()
        self.state.preview_after_id = "after#1"
        gui.update_file_widget()
        self.state.window.after_cancel.assert_called_once_with("after#1")
        self.state.file_widget.refresh_pipeline.assert_called_once()
        self.assertIsNone(self.state.preview_after_id)

    def test_unreadable_card_is_omitted_without_hiding_valid_cards(self):
        view = object.__new__(gui.FileWidget)
        view.file_list = ["good.png", "bad.mp4", "another.png"]
        view.unreadable_files = []
        view.ui_components = []
        view._create_file_card = Mock(side_effect=[{}, ValueError("unreadable"), {}])
        view._render_cards()
        self.assertEqual(view.file_list, ["good.png", "another.png"])
        self.assertEqual(view.unreadable_files, ["bad.mp4"])
        self.assertEqual(len(view.ui_components), 2)

    def test_refresh_cancels_progress_animations_before_destroying_sections(self):
        view = object.__new__(gui.FileWidget)
        bar, section, content = Mock(), Mock(), Mock()
        view._animation_ids = {"after#1": bar}
        view.ui_components = [dict(dynamic_section=section, content=content,
                                   dynamic_row=2, is_video=True, file_path="clip.mp4",
                                   width=1920, height=1080)]
        view._create_dynamic_section = Mock(return_value=Mock())
        view.refresh_pipeline()
        bar.after_cancel.assert_called_once_with("after#1")
        section.destroy.assert_called_once()
        self.assertFalse(view._animation_ids)

    def test_video_preview_releases_capture_on_decode_failure(self):
        view = object.__new__(gui.FileWidget)
        capture = Mock()
        capture.read.side_effect = ValueError("decode")
        with patch.object(gui, "opencv_VideoCapture", return_value=capture):
            with self.assertRaises(ValueError):
                view.extract_file_icon("clip.mp4")
        capture.release.assert_called_once()

    def test_active_highlight_matches_one_based_processing_order(self):
        view = object.__new__(gui.FileWidget)
        cards = [Mock(), Mock()]
        view.ui_components = [dict(card=card) for card in cards]
        view.highlight_active_file(2)
        cards[0].configure.assert_called_with(border_color=gui.CARD_BORDER_COLOR)
        cards[1].configure.assert_called_with(border_color=gui.CARD_ACCENT_COLOR)
        view.clear_active_highlight()
        cards[1].configure.assert_called_with(border_color=gui.CARD_BORDER_COLOR)

    def test_removing_file_resets_previous_video_rows(self):
        view = object.__new__(gui.FileWidget)
        view.file_list = ["clip.mp4", "image.png"]
        view.clean_file_list = Mock()
        view._render_cards = Mock()
        self.state.status_bar = Mock()
        view._remove_file("clip.mp4")
        self.assertEqual(self.state.selected_file_list, ["image.png"])
        self.state.status_bar.set_video_progress.assert_called_once_with(None)
        self.state.info_message.set.assert_called_once_with("Ready — 1 input file(s)")


class LifecycleTests(StateTestCase):
    def test_manager_shuts_down_if_window_creation_fails(self):
        manager = Mock()
        manager.Queue.return_value = Queue()
        manager.Event.return_value = Event()
        with patch.object(gui, "multiprocessing_Manager", return_value=manager), \
                patch.object(gui, "load_user_preferences", return_value=gui.UserPreferences()), \
                patch.object(gui, "set_appearance_mode"), \
                patch.object(gui, "set_default_color_theme"), \
                patch.object(gui, "apply_app_zoom"), \
                patch.object(gui, "allow_sleep"), \
                patch.object(gui, "FileDropWindow", side_effect=RuntimeError("Tk init failed")):
            with self.assertRaisesRegex(RuntimeError, "Tk init failed"):
                gui.main()
        manager.shutdown.assert_called_once()

    def test_close_stops_before_destroy_and_cancels_callbacks_even_if_save_fails(self):
        calls = []
        window = self.state.window
        self.state.after_ids = {"after#1"}
        self.state.file_widget = Mock()
        window.destroy.side_effect = lambda: calls.append("destroy")
        with patch.object(gui, "save_user_choices_in_json", side_effect=OSError("readonly")), \
                patch.object(gui, "stop_button_command", side_effect=lambda: calls.append("stop")):
            gui.on_app_close()
        self.assertEqual(calls, ["stop", "destroy"])
        self.assertTrue(self.state.closing)
        window.after_cancel.assert_called_once_with("after#1")
        self.assertFalse(self.state.after_ids)

    def test_destroy_window_is_idempotent_after_tk_already_destroyed_root(self):
        window = self.state.window
        widget = Mock()
        self.state.file_widget = widget
        widget.destroy.side_effect = gui.TclError("application has been destroyed")
        window.grab_release.side_effect = gui.TclError("application has been destroyed")
        window.destroy.side_effect = gui.TclError("application has been destroyed")
        gui._destroy_window()
        gui._destroy_window()
        self.assertIsNone(self.state.window)
        self.assertIsNone(self.state.file_widget)
        window.destroy.assert_called_once()
        widget.destroy.assert_called_once()

    @unittest.skipUnless(sys.platform == "win32" or os.environ.get("DISPLAY"),
                         "Real Tk smoke requires a desktop display")
    def test_real_mainloop_monkeypatch_uses_native_drop_root_and_renders_cards(self):
        from PIL import Image
        manager = Mock()
        manager.Queue.return_value = Queue()
        manager.Event.return_value = Event()
        with TemporaryDirectory() as directory:
            image = Path(directory) / "résumé {one}.PNG"
            Image.new("RGB", (64, 48)).save(image)

            def smoke(window):
                self.assertIsInstance(window, gui.FileDropWindow)
                self.assertIsNotNone(window.file_drop_target)
                self.assertTrue(gui.select_input_files([image]))
                self.assertEqual(len(gui.app_state.file_widget.ui_components), 1)
                self.assertEqual(gui.app_state.file_widget.get_selected_file_list(), [str(image.resolve())])
                window.update()
                self.assertTrue(gui.clear_input_files())
                self.assertTrue(gui.app_state.input_widgets)

            with patch.object(gui, "multiprocessing_Manager", return_value=manager), \
                    patch.object(gui, "load_user_preferences", return_value=gui.UserPreferences()), \
                    patch.object(gui.GPU, "detect"), \
                    patch.object(gui.GPU, "menu_list", return_value=["No GPU found"]), \
                    patch.object(gui.GPU, "vram_for", return_value=None), \
                    patch.object(gui.GPU, "codec_for", return_value=None), \
                    patch.object(gui, "get_AI_engine_info", return_value="QualityScaler UI test"), \
                    patch.object(gui, "allow_sleep"), \
                    patch("customtkinter.CTk.mainloop", smoke):
                gui.main()
        manager.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
