"""DPI policy/API contracts without real monitor or process-awareness changes."""

import builtins
import importlib.util
from pathlib import Path
from queue import Queue
import sys
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_dpi():
    spec = importlib.util.spec_from_file_location("dpi_under_test", ROOT / "qualityscaler/dpi.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DpiPolicyTests(unittest.TestCase):
    def setUp(self):
        self.dpi = load_dpi()
        self.per_monitor = Mock(return_value=0)
        self.system = Mock(return_value=1)
        self.ctypes = SimpleNamespace(windll=SimpleNamespace(
            shcore=SimpleNamespace(SetProcessDpiAwareness=self.per_monitor),
            user32=SimpleNamespace(SetProcessDPIAware=self.system),
        ))
        for patcher in (
            patch.object(sys, "platform", "win32"),
            patch.dict(sys.modules, {"ctypes": self.ctypes}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_per_monitor_v1_matches_ctk_fixed_dialog_compatibility_policy(self):
        self.dpi._enable_windows_dpi_awareness()
        self.per_monitor.assert_called_once_with(2)
        self.system.assert_not_called()

    def test_existing_manifest_or_host_awareness_is_not_overridden(self):
        for denied in (-2147024891, 0x80070005):
            with self.subTest(hresult=denied):
                self.per_monitor.reset_mock()
                self.per_monitor.return_value = denied
                self.dpi = load_dpi()
                self.dpi._enable_windows_dpi_awareness()
                self.per_monitor.assert_called_once_with(2)
                self.system.assert_not_called()

    def test_missing_per_monitor_api_falls_back_to_system_awareness(self):
        del self.ctypes.windll.shcore
        self.dpi._enable_windows_dpi_awareness()
        self.system.assert_called_once_with()

    def test_failed_per_monitor_api_falls_back_without_startup_error(self):
        for failure in (OSError("API unavailable"), AttributeError("missing API"), -2147024809):
            with self.subTest(failure=failure):
                self.dpi = load_dpi()
                self.per_monitor.side_effect = failure if isinstance(failure, Exception) else None
                self.per_monitor.return_value = failure
                self.system.reset_mock()
                self.dpi._enable_windows_dpi_awareness()
                self.system.assert_called_once_with()

    def test_unsupported_windows_apis_do_not_fail_startup(self):
        del self.ctypes.windll.shcore
        for failure in (OSError("unsupported"), AttributeError("missing")):
            with self.subTest(failure=failure):
                self.dpi = load_dpi()
                self.system.side_effect = failure
                self.dpi._enable_windows_dpi_awareness()
        with patch.dict(sys.modules, {"ctypes": None}):
            load_dpi()._enable_windows_dpi_awareness()

    def test_awareness_is_not_retried_for_later_toplevel_windows(self):
        self.dpi._enable_windows_dpi_awareness()
        self.dpi._enable_windows_dpi_awareness()
        self.per_monitor.assert_called_once_with(2)
        with patch.object(self.dpi.os, "getpid", return_value=987654321):
            self.dpi._enable_windows_dpi_awareness()
        self.assertEqual(self.per_monitor.call_count, 2)

    def test_ctk_monitor_detection_is_reused_without_user_scale_factors(self):
        detect = Mock(return_value=1.75)
        tracker = SimpleNamespace(
            get_window_dpi_scaling=detect,
            deactivate_automatic_dpi_awareness=False,
            widget_scaling=1, window_scaling=1,
        )
        with patch.dict(sys.modules, {"customtkinter": SimpleNamespace(ScalingTracker=tracker)}):
            self.dpi.configure_high_dpi()
            detector = tracker.get_window_dpi_scaling
            self.dpi.configure_high_dpi()
            self.assertIs(tracker.get_window_dpi_scaling, detector)
        # Use the installed hooks as the library does for roots/toplevels.
        tracker.activate_high_dpi_awareness()
        root = object()
        self.assertEqual(tracker.get_window_dpi_scaling(root), 1.75)
        detect.assert_called_once_with(root)
        self.per_monitor.assert_called_once_with(2)
        self.assertFalse(tracker.deactivate_automatic_dpi_awareness)
        self.assertEqual((tracker.widget_scaling, tracker.window_scaling), (1, 1))

    def test_ctk_missing_monitor_api_retains_library_default_factor(self):
        for failure in (AttributeError("GetDpiForMonitor"), OSError("unsupported")):
            with self.subTest(failure=failure):
                self.dpi = load_dpi()
                tracker = SimpleNamespace(get_window_dpi_scaling=Mock(side_effect=failure))
                with patch.dict(sys.modules, {"customtkinter": SimpleNamespace(ScalingTracker=tracker)}):
                    self.dpi.configure_high_dpi()
                self.assertEqual(tracker.get_window_dpi_scaling(object()), 1)


class DpiIsolationTests(unittest.TestCase):
    def test_import_and_portable_paths_have_no_windows_or_gui_dependencies(self):
        original_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"ctypes", "tkinter", "customtkinter"}:
                raise AssertionError("Unexpected dependency: " + name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guard):
            dpi = load_dpi()
            for platform in ("linux", "darwin"):
                with self.subTest(platform=platform), patch.object(sys, "platform", platform):
                    dpi.configure_high_dpi()
                    dpi._enable_windows_dpi_awareness()
        self.assertIsNone(dpi._awareness_pid)

    def test_desktop_configures_awareness_before_any_tk_window(self):
        from qualityscaler import gui

        calls = []
        manager = Mock()
        manager.Queue.return_value = Queue()
        manager.Event.return_value = Event()

        def create_window():
            calls.append("Tk")
            raise RuntimeError("stop before Tk")

        with patch.object(gui, "configure_high_dpi", side_effect=lambda: calls.append("DPI")), \
                patch.object(gui, "load_user_preferences", return_value=gui.UserPreferences()), \
                patch.object(gui, "set_appearance_mode"), \
                patch.object(gui, "set_default_color_theme"), \
                patch.object(gui, "multiprocessing_Manager", return_value=manager), \
                patch.object(gui, "allow_sleep"), \
                patch.object(gui, "FileDropWindow", side_effect=create_window):
            with self.assertRaisesRegex(RuntimeError, "stop before Tk"):
                gui.main()
        self.assertEqual(calls, ["DPI", "Tk"])
        manager.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
