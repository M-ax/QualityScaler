"""Native splash/launcher contracts without Tk, PyInstaller, or GPU execution."""

import builtins
import importlib.util
import multiprocessing
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_CODE = compile(
    (ROOT / "QualityScaler.py").read_text(encoding="utf-8"),
    str(ROOT / "QualityScaler.py"),
    "exec",
)


def load_startup():
    # Fresh module state per test, without reload of the real pyi_splash IPC.
    spec = importlib.util.spec_from_file_location(
        "startup_under_test", ROOT / "qualityscaler/startup.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker_splash_probe(connection):
    """A real spawned worker must ignore even an inherited, nonzero parent port."""
    try:
        sys.frozen = True
        os.environ["_PYI_SPLASH_IPC"] = "47631"
        from qualityscaler import startup

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"pyi_splash", "tkinter", "customtkinter", "onnxruntime"}:
                raise AssertionError("Worker imported splash/GUI/runtime: " + name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            startup._enable_splash()
            startup.update_splash("Worker must not update parent")
            startup.close_splash()
        connection.send(("ok", startup._owner_pid,
                         os.environ.get("_PYI_SPLASH_IPC")))
    except BaseException as error:
        connection.send(("error", repr(error)))
    finally:
        connection.close()


class SplashHelperTests(unittest.TestCase):
    def setUp(self):
        self.startup = load_startup()
        self.splash = ModuleType("pyi_splash")
        self.splash.is_alive = Mock(return_value=True)
        self.splash.update_text = Mock()
        self.splash.close = Mock()
        patches = (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "argv", ["QualityScaler.exe"]),
            patch.dict(os.environ, {"_PYI_SPLASH_IPC": "47631"}),
            patch.dict(sys.modules, {"pyi_splash": self.splash}),
            patch.object(self.startup, "current_process",
                         return_value=SimpleNamespace(name="MainProcess")),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_no_ipc(self):
        self.splash.is_alive.assert_not_called()
        self.splash.update_text.assert_not_called()
        self.splash.close.assert_not_called()

    def test_source_and_unactivated_frozen_calls_are_noops_without_imports(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"pyi_splash", "tkinter", "customtkinter", "onnxruntime"}:
                raise AssertionError("No-op imported optional dependency: " + name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            # Frozen CLI/self-test code importing GUI helpers never opts in.
            self.startup.update_splash("Not the desktop launcher")
            self.startup.close_splash()
            with patch.object(sys, "frozen", False):
                self.startup._enable_splash()
                self.startup.update_splash("Source run")
                self.startup.close_splash()
        self.assertIsNone(self.startup._owner_pid)
        self.assert_no_ipc()

    def test_importing_helper_itself_is_lazy_and_gui_free(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"pyi_splash", "tkinter", "customtkinter", "onnxruntime"}:
                raise AssertionError("Helper import loaded optional dependency: " + name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            module = load_startup()
            module._enable_splash()
        self.assertIsNone(module._splash)
        self.assert_no_ipc()

    def test_updates_and_close_share_one_lazy_connection_consuming_ipc_variable(self):
        self.startup._enable_splash()
        self.assert_no_ipc()
        self.startup.update_splash("Loading interface...")
        # PyInstaller 6.16 deletes this variable on the first pyi_splash import.
        os.environ.pop("_PYI_SPLASH_IPC")
        self.startup.update_splash("Preparing desktop...")
        self.startup.close_splash()
        self.startup.close_splash()
        self.startup._enable_splash()
        self.startup.update_splash("Must not reopen")
        self.assertEqual(
            [call.args[0] for call in self.splash.update_text.call_args_list],
            ["Loading interface...", "Preparing desktop..."],
        )
        self.splash.close.assert_called_once_with()

    def test_close_before_first_update_is_lazy_and_final(self):
        self.startup._enable_splash()
        self.startup.close_splash()
        self.startup.update_splash("Too late")
        self.startup.close_splash()
        self.splash.update_text.assert_not_called()
        self.splash.close.assert_called_once_with()

    def test_phase_progress_uses_existing_text_channel_and_is_bounded(self):
        self.startup._enable_splash()
        self.startup.update_splash("Loading...", progress=72)
        self.startup.update_splash("Starting...", progress=-5)
        self.startup.update_splash("Ready", progress=120)
        self.assertEqual(
            [call.args[0] for call in self.splash.update_text.call_args_list],
            ["QS_PROGRESS:72:Loading...", "QS_PROGRESS:0:Starting...",
             "QS_PROGRESS:100:Ready"],
        )

    def test_missing_disabled_or_invalid_port_never_imports_pyi_splash(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "pyi_splash":
                raise AssertionError("Unavailable IPC must not import pyi_splash")
            return original_import(name, *args, **kwargs)

        for port in (None, "", "0", "not-a-port", "-1", "65536"):
            with self.subTest(port=port):
                startup = load_startup()
                if port is None:
                    os.environ.pop("_PYI_SPLASH_IPC", None)
                else:
                    os.environ["_PYI_SPLASH_IPC"] = port
                with patch("builtins.__import__", side_effect=guarded_import):
                    startup._enable_splash()
                    startup.update_splash("Unavailable")
                    startup.close_splash()
        self.assert_no_ipc()

    def test_missing_optional_module_is_not_a_startup_failure(self):
        with patch.dict(sys.modules, {"pyi_splash": None}):
            self.startup._enable_splash()
            self.startup.update_splash("Not bundled")
            self.startup.close_splash()
        # No subsequent retry or reopening when the optional module appears.
        self.startup.update_splash("Still disabled")
        self.assert_no_ipc()

    def test_ipc_import_failure_is_optional_and_not_retried(self):
        original_import = builtins.__import__

        def fail_splash_import(name, *args, **kwargs):
            if name == "pyi_splash":
                raise OSError("IPC unavailable during import")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_splash_import):
            self.startup._enable_splash()
            self.startup.update_splash("Loading")
            self.startup.close_splash()
        self.startup.update_splash("Do not reconnect")
        self.assert_no_ipc()

    def test_uninitialized_or_already_closed_ipc_is_a_noop(self):
        self.splash.is_alive.return_value = False
        self.startup._enable_splash()
        self.startup.update_splash("No live connection")
        self.startup.close_splash()
        self.splash.update_text.assert_not_called()
        self.splash.close.assert_called_once_with()

    def test_ipc_update_failure_closes_once_without_masking_startup(self):
        for operation in ("is_alive", "update_text"):
            for error in (ConnectionError("lost IPC"), RuntimeError("not initialized")):
                with self.subTest(operation=operation, error=type(error).__name__):
                    self.startup = load_startup()
                    self.splash.is_alive.reset_mock(side_effect=True)
                    self.splash.update_text.reset_mock(side_effect=True)
                    self.splash.close.reset_mock()
                    getattr(self.splash, operation).side_effect = error
                    self.startup._enable_splash()
                    self.startup.update_splash("Loading")
                    self.startup.close_splash()
                    self.startup.update_splash("Must not retry")
                    self.splash.close.assert_called_once_with()

    def test_close_failure_is_harmless_and_does_not_reconnect(self):
        self.splash.close.side_effect = OSError("connection gone")
        self.startup._enable_splash()
        self.startup.close_splash()
        self.startup.close_splash()
        self.startup.update_splash("Must not reopen")
        self.splash.close.assert_called_once_with()
        self.splash.update_text.assert_not_called()

    def test_worker_cannot_opt_in_even_with_inherited_parent_ipc(self):
        with patch.object(self.startup, "current_process",
                          return_value=SimpleNamespace(name="SpawnProcess-1")):
            self.startup._enable_splash()
            self.startup.update_splash("Worker")
            self.startup.close_splash()
        self.assert_no_ipc()
        self.assertIsNone(self.startup._owner_pid)

    def test_self_test_and_spawn_bootstrap_arguments_cannot_opt_in(self):
        for args in (
            ["--self-test", "report.json"],
            ["--multiprocessing-fork", "parent_pid=123"],
            ["-c", "from multiprocessing.resource_tracker import main; main(7)"],
        ):
            with self.subTest(args=args), patch.object(
                sys, "argv", ["QualityScaler.exe", *args],
            ):
                self.startup._enable_splash()
                self.startup.update_splash("Not desktop")
                self.startup.close_splash()
                self.assertIsNone(self.startup._owner_pid)
        self.assert_no_ipc()

    def test_inherited_open_connection_is_not_used_or_closed_by_another_pid(self):
        self.startup._enable_splash()
        self.startup.update_splash("Parent")
        with patch.object(self.startup.os, "getpid", return_value=os.getpid() + 1):
            self.startup.update_splash("Child")
            self.startup.close_splash()
            self.startup._enable_splash()
        self.splash.update_text.assert_called_once_with("Parent")
        self.splash.close.assert_not_called()
        self.startup.close_splash()
        self.splash.close.assert_called_once_with()


class LauncherSplashTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.startup = ModuleType("qualityscaler.startup")
        self.startup._enable_splash = Mock(
            side_effect=lambda: self.events.append("enable"),
        )
        self.startup.update_splash = Mock(
            side_effect=lambda text, **kwargs: self.events.append(text),
        )
        self.startup.close_splash = Mock(
            side_effect=lambda: self.events.append("close"),
        )
        self.gui = ModuleType("qualityscaler.gui")
        self.gui.main = Mock(side_effect=lambda: self.events.append("desktop"))
        self.self_test = ModuleType("qualityscaler.self_test")
        self.self_test.run_self_test = Mock(return_value=0)
        patches = (
            patch.object(sys, "argv", ["QualityScaler.exe"]),
            patch.dict(sys.modules, {
                "qualityscaler.startup": self.startup,
                "qualityscaler.gui": self.gui,
                "qualityscaler.self_test": self.self_test,
            }),
            patch.object(
                multiprocessing, "freeze_support",
                side_effect=lambda: self.events.append("freeze_support"),
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_launcher(self, name="__main__"):
        exec(LAUNCHER_CODE, {"__name__": name})

    def test_freeze_support_precedes_splash_and_gui_import(self):
        original_import = builtins.__import__

        def record_import(name, *args, **kwargs):
            if name in {"qualityscaler.startup", "qualityscaler.gui"}:
                self.events.append(name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=record_import):
            self.run_launcher()
        self.assertEqual(self.events, [
            "freeze_support", "qualityscaler.startup", "enable",
            "Loading QualityScaler...", "qualityscaler.gui",
            "Preparing desktop...", "desktop", "close",
        ])
        self.assertEqual(
            [call.kwargs for call in self.startup.update_splash.call_args_list],
            [{"progress": 72}, {"progress": 82}],
        )

    def test_gui_import_error_closes_splash_and_propagates_original_error(self):
        original_import = builtins.__import__
        error = ImportError("broken desktop dependency")

        def fail_gui_import(name, *args, **kwargs):
            if name == "qualityscaler.gui":
                raise error
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_gui_import):
            with self.assertRaises(ImportError) as caught:
                self.run_launcher()
        self.assertIs(caught.exception, error)
        self.startup.close_splash.assert_called_once_with()
        self.gui.main.assert_not_called()

    def test_all_desktop_exit_paths_close_without_masking_original_error(self):
        for error in (RuntimeError("UI failed"), SystemExit(2), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.startup.close_splash.reset_mock()
                self.gui.main.side_effect = error
                with self.assertRaises(type(error)) as caught:
                    self.run_launcher()
                self.assertIs(caught.exception, error)
                self.startup.close_splash.assert_called_once_with()

    def test_self_test_never_activates_or_closes_a_parent_splash(self):
        for args in (["--self-test", "report.json"], ["--self-test"]):
            with self.subTest(args=args), patch.object(
                sys, "argv", ["QualityScaler.exe", *args],
            ):
                with self.assertRaises(SystemExit):
                    self.run_launcher()
                self.startup._enable_splash.assert_not_called()
                self.startup.update_splash.assert_not_called()
                self.startup.close_splash.assert_not_called()
                self.gui.main.assert_not_called()
        self.self_test.run_self_test.assert_called_once_with("report.json")

    def test_multiprocessing_bootstrap_exits_before_splash_or_desktop(self):
        with patch.object(multiprocessing, "freeze_support", side_effect=SystemExit(0)):
            with self.assertRaises(SystemExit):
                self.run_launcher()
        self.startup._enable_splash.assert_not_called()
        self.startup.close_splash.assert_not_called()
        self.gui.main.assert_not_called()

    def test_spawn_reimport_does_not_run_any_launcher_code(self):
        self.run_launcher(name="__mp_main__")
        self.assertEqual(self.events, [])


class SpawnIsolationTests(unittest.TestCase):
    def test_real_spawn_worker_cannot_import_or_use_parent_splash(self):
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=worker_splash_probe, args=(writer,))
        try:
            process.start()
            writer.close()
            self.assertTrue(reader.poll(15), "Spawn worker did not return")
            result = reader.recv()
            process.join(timeout=15)
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(result, ("ok", None, "47631"))
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            reader.close()
            writer.close()


if __name__ == "__main__":
    unittest.main()
