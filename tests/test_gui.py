from inspect import signature
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
from qualityscaler.jobs import upscale_orchestrator


class DesktopBackendWiringTests(unittest.TestCase):
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
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_start_uses_modular_orchestrator_with_current_signature(self):
        process = Mock()
        status = Queue()
        output = Mock()
        output.get.return_value = "output directory"
        event = Event()
        values = {
            "process_status_q": status, "event_stop_upscale_process": event,
            "selected_file_list": ["clip.mp4"], "selected_AI_model": "Testx2",
            "selected_gpu": "Auto", "selected_keep_frames": True,
            "selected_AI_multithreading": 2, "selected_blending_factor": 0.5,
            "selected_image_extension": ".png", "selected_video_extension": ".mp4",
            "selected_video_codec": "x264", "tiles_resolution": 1024,
            "input_resize_factor": 1.0, "output_resize_factor": 1.0,
            "selected_output_path": output, "info_message": Mock(),
        }
        with patch.multiple(gui, create=True, **values), \
                patch.object(gui, "user_input_checks", return_value=True), \
                patch.object(gui, "place_stop_button"), \
                patch.object(gui, "Process", return_value=process) as constructor, \
                patch.object(gui, "Thread"):
            gui.upscale_button_command()
        self.assertIs(constructor.call_args.kwargs["target"], upscale_orchestrator)
        bound = signature(upscale_orchestrator).bind(
            *constructor.call_args.kwargs["args"]
        )
        self.assertIs(bound.arguments["event_stop_upscale_process"], event)
        self.assertEqual(bound.arguments["selected_file_list"], ["clip.mp4"])
        process.start.assert_called_once()

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
            [sys.executable, "-c", script, str(root)],
            capture_output=True, text=True, check=False, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stop_waits_for_checkpoint_before_hard_kill(self):
        process = Mock(pid=123)
        process.is_alive.return_value = False
        event = Event()
        status = Queue()
        status.put("old progress")
        with patch.multiple(
            gui, create=True, process_upscale_orchestrator=process,
            event_stop_upscale_process=event, process_status_q=status,
        ):
            gui.stop_upscale_process()
        process.join.assert_called_once_with(timeout=10)
        process.kill.assert_not_called()
        self.assertTrue(status.empty())
        self.assertFalse(event.is_set())

    def test_hung_worker_fallback_kills_ffmpeg_descendants_and_joins(self):
        process = Mock(pid=123)
        process.is_alive.return_value = True
        child = Mock()
        event = Event()
        with patch.multiple(
            gui, create=True, process_upscale_orchestrator=process,
            event_stop_upscale_process=event, process_status_q=Queue(),
        ), patch.object(gui, "psutil_Process") as lookup:
            lookup.return_value.children.return_value = [child]
            gui.stop_upscale_process()
        child.kill.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.join.call_count, 2)
        self.assertFalse(event.is_set())


class InputSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.first = self.root / "résumé {one}.PNG"
        self.second = self.root / "second clip.MP4"
        self.first.write_bytes(b"preview is mocked")
        self.second.write_bytes(b"preview is mocked")
        self.views = []
        self.reject = set()
        self.status = Mock()
        self.placeholder = (Mock(), Mock(), Mock())
        window = Mock()
        window.winfo_rootx.return_value = 100
        window.winfo_rooty.return_value = 200
        window.winfo_width.return_value = 1000
        window.winfo_height.return_value = 675
        patcher = patch.multiple(
            gui, create=True, file_widget=None, input_widgets=self.placeholder,
            selected_file_list=[], process_upscale_orchestrator=None,
            window=window, info_message=self.status,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
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
        self.assertEqual(gui.selected_file_list,
                         [str(self.first.resolve()), str(self.second.resolve())])
        first_view.destroy.assert_called_once()
        for widget in self.placeholder:
            widget.destroy.assert_called_once()
        self.assertTrue(gui.select_input_files((str(self.first),)))
        self.assertEqual(self.constructor.call_count, 2)

    def test_picker_cancellation_leaves_selection_and_status_untouched(self):
        gui.select_input_files([self.first])
        view = gui.file_widget
        self.status.reset_mock()
        with patch.object(gui.filedialog, "askopenfilenames", return_value=()):
            gui.open_files_action()
        self.assertIs(gui.file_widget, view)
        view.destroy.assert_not_called()
        self.status.set.assert_not_called()

    def test_invalid_drop_leaves_existing_selection_untouched(self):
        gui.select_input_files([self.first])
        view = gui.file_widget
        notes = self.root / "notes.txt"
        notes.write_text("ignore")
        self.assertFalse(gui.select_input_files([notes, self.root, "missing.jpg"]))
        self.assertIs(gui.file_widget, view)
        self.assertEqual(gui.selected_file_list, [str(self.first.resolve())])
        view.destroy.assert_not_called()

    def test_unreadable_new_media_does_not_replace_a_valid_selection(self):
        gui.select_input_files([self.first])
        view = gui.file_widget
        self.reject.add(str(self.second.resolve()))
        self.assertFalse(gui.select_input_files([self.second]))
        self.assertIs(gui.file_widget, view)
        view.destroy.assert_not_called()
        self.views[-1].destroy.assert_called_once()

    def test_active_job_refuses_picker_drop_and_clear(self):
        gui.select_input_files([self.first])
        view = gui.file_widget
        gui.process_upscale_orchestrator = SimpleNamespace(is_alive=lambda: True)
        with patch.object(gui.filedialog, "askopenfilenames") as picker:
            gui.open_files_action()
        picker.assert_not_called()
        self.assertFalse(gui.select_input_files([self.second]))
        self.assertFalse(gui.clear_input_files())
        self.assertIs(gui.file_widget, view)
        view.destroy.assert_not_called()

    def test_drop_only_accepts_the_left_input_pane(self):
        self.assertTrue(gui.accept_input_drop(SimpleNamespace(x_root=350, y_root=300)))
        for x, y in ((850, 300), (99, 300), (350, 199), (350, 875)):
            with self.subTest(x=x, y=y):
                self.assertFalse(gui.accept_input_drop(SimpleNamespace(x_root=x, y_root=y)))

    def test_clear_resets_widget_and_queue_before_showing_placeholder(self):
        gui.select_input_files([self.first])
        view = gui.file_widget
        with patch.object(gui, "place_loadFile_section") as placeholder:
            self.assertTrue(gui.clear_input_files())
        self.assertIsNone(gui.file_widget)
        self.assertEqual(gui.selected_file_list, [])
        view.destroy.assert_called_once()
        placeholder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
