from contextlib import redirect_stdout
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qualityscaler import platform_support


class PlatformImportTests(unittest.TestCase):
    def test_import_is_gui_free_and_windows_dependencies_are_lazy(self):
        script = """
import builtins
import sys
original_import = builtins.__import__
blocked = {'customtkinter', 'tkinter', 'onnxruntime', 'QualityScaler',
           'ctypes', 'winotify', 'win32api', 'win32con', 'pythoncom'}
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('Unexpected platform dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import qualityscaler.platform_support
assert not blocked.intersection(sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_logo_resolves_against_repository_root(self):
        with patch.dict(sys.__dict__):
            sys.__dict__.pop("_MEIPASS", None)
            importlib.reload(platform_support)
        try:
            self.assertEqual(
                platform_support.LOGO_PNG_PATH,
                str(Path(__file__).resolve().parents[1] / "Assets" / "logo.png"),
            )
        finally:
            importlib.reload(platform_support)

    def test_logo_resolves_against_pyinstaller_bundle(self):
        bundle = os.path.abspath("bundled resources")
        try:
            with patch.object(sys, "_MEIPASS", bundle, create=True):
                importlib.reload(platform_support)
                self.assertEqual(
                    platform_support.LOGO_PNG_PATH,
                    os.path.join(bundle, "Assets", "logo.png"),
                )
        finally:
            importlib.reload(platform_support)


class SleepTests(unittest.TestCase):
    def test_non_windows_helpers_do_not_import_or_call_platform_tools(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                with patch.object(sys, "platform", platform):
                    with patch.dict(sys.modules, {"ctypes": None, "winotify": None}):
                        with patch("builtins.print") as log:
                            platform_support.prevent_sleep()
                            platform_support.allow_sleep()
                            platform_support.show_completion_notification()
                log.assert_not_called()

    def test_windows_execution_state_keeps_system_but_not_display_awake(self):
        execution_state = Mock()
        ctypes = SimpleNamespace(
            windll=SimpleNamespace(
                kernel32=SimpleNamespace(SetThreadExecutionState=execution_state)
            )
        )
        with patch.object(sys, "platform", "win32"):
            with patch.dict(sys.modules, {"ctypes": ctypes}):
                platform_support.prevent_sleep()
                execution_state.assert_called_once_with(0x80000001)
                execution_state.reset_mock()
                platform_support.allow_sleep()
                execution_state.assert_called_once_with(0x80000000)

    def test_windows_execution_state_errors_are_best_effort(self):
        execution_state = Mock(side_effect=OSError("API unavailable"))
        ctypes = SimpleNamespace(
            windll=SimpleNamespace(
                kernel32=SimpleNamespace(SetThreadExecutionState=execution_state)
            )
        )
        with patch.object(sys, "platform", "win32"):
            with patch.dict(sys.modules, {"ctypes": ctypes}):
                with redirect_stdout(io.StringIO()) as output:
                    platform_support.prevent_sleep()
                    platform_support.allow_sleep()
        self.assertIn("could not prevent system sleep", output.getvalue())
        self.assertIn("could not restore system sleep state", output.getvalue())

    def test_missing_ctypes_is_best_effort(self):
        with patch.object(sys, "platform", "win32"):
            with patch.dict(sys.modules, {"ctypes": None}):
                with redirect_stdout(io.StringIO()) as output:
                    platform_support.prevent_sleep()
                    platform_support.allow_sleep()
        self.assertIn("could not prevent system sleep", output.getvalue())
        self.assertIn("could not restore system sleep state", output.getvalue())


class CompletionNotificationTests(unittest.TestCase):
    def test_windows_notification_uses_bundled_logo_and_upstream_message(self):
        notification = Mock()
        with patch.object(sys, "platform", "win32"):
            with patch.dict(
                sys.modules, {"winotify": SimpleNamespace(Notification=notification)}
            ):
                platform_support.show_completion_notification()
        notification.assert_called_once_with(
            app_id="QualityScaler",
            title="QualityScaler",
            msg="All files completed! :)",
            icon=platform_support.LOGO_PNG_PATH,
        )
        notification.return_value.show.assert_called_once_with()

    def test_missing_notification_provider_is_best_effort(self):
        with patch.object(sys, "platform", "win32"):
            with patch.dict(sys.modules, {"winotify": None}):
                with redirect_stdout(io.StringIO()) as output:
                    platform_support.show_completion_notification()
        self.assertIn("could not show system notification", output.getvalue())

    def test_notification_constructor_and_show_errors_are_best_effort(self):
        for operation in ("constructor", "show"):
            with self.subTest(operation=operation):
                notification = Mock()
                if operation == "constructor":
                    notification.side_effect = OSError("toast unavailable")
                else:
                    notification.return_value.show.side_effect = OSError(
                        "toast unavailable"
                    )
                with patch.object(sys, "platform", "win32"):
                    with patch.dict(
                        sys.modules,
                        {"winotify": SimpleNamespace(Notification=notification)},
                    ):
                        with redirect_stdout(io.StringIO()) as output:
                            platform_support.show_completion_notification()
                self.assertIn("could not show system notification", output.getvalue())


if __name__ == "__main__":
    unittest.main()
