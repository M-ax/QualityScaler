"""GPU-free completion/process teardown contracts using real Tcl timers."""

from multiprocessing import get_context
from queue import Queue
from threading import Event, current_thread, get_ident
from time import monotonic, sleep
from tkinter import Tcl
import unittest
from unittest.mock import Mock, call, patch

from qualityscaler import gui


def terminal_then_wait(status_queue, ready, release):
    """A real spawn worker whose terminal signal intentionally precedes exit."""
    status_queue.put("Completed")
    ready.set()
    if not release.wait(15):
        raise RuntimeError("Test did not release the worker")


def pump_until(window, predicate, timeout=5):
    deadline = monotonic() + timeout
    while not predicate():
        if monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Tcl callbacks")
        window.update()
        sleep(0.001)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        # Tcl's actual after/update event loop works without a desktop or Tk
        # widgets. No layout, inference session, GPU, or OS toast is created.
        self.window = Tcl()
        self.process = Mock(pid=123, exitcode=0)
        self.process.is_alive.return_value = True
        self.state = gui.AppState(
            preferences=gui.UserPreferences(), window=self.window,
            info_message=Mock(), file_widget=Mock(),
            process_upscale_orchestrator=self.process,
            process_status_q=Queue(), event_stop_upscale_process=Event(),
            processing=True,
        )
        self.patch_value("app_state", self.state)
        self.refresh = self.patch_value("update_file_widget")
        self.allow_sleep = self.patch_value("allow_sleep")
        self.error = self.patch_value("show_error_message")
        self.thread = self.patch_value("Thread")
        patcher = patch.object(gui.App, "place_upscale_button")
        self.idle_button = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.cancel_callbacks)

    def patch_value(self, name, value=None):
        patcher = patch.object(gui, name, Mock() if value is None else value)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def cancel_callbacks(self):
        for handle in tuple(self.state.after_ids):
            self.window.after_cancel(handle)
        self.state.after_ids.clear()

    def send(self, message):
        self.state.process_status_q.put(message)
        gui.check_upscale_steps()

    def test_early_completion_keeps_busy_and_polls_without_join_stop_or_toast(self):
        self.send(gui.COMPLETED_STATUS)
        self.assertEqual(self.state.pending_upscale_status, gui.COMPLETED_STATUS)
        self.state.info_message.set.assert_called_with(
            "Finalizing — waiting for worker cleanup..."
        )
        self.assertTrue(gui.inputs_are_busy())
        self.assertFalse(self.state.stopping)
        self.assertFalse(self.state.event_stop_upscale_process.is_set())
        self.assertIsNone(self.state.stop_deadline)
        self.assertIsNotNone(self.state.poll_after_id)
        self.assertEqual(len(self.state.after_ids), 1)
        self.process.join.assert_not_called()
        self.process.kill.assert_not_called()
        self.refresh.assert_not_called()
        self.allow_sleep.assert_not_called()
        self.thread.assert_not_called()
        self.idle_button.assert_not_called()

    def test_success_only_after_exit_reaps_nonblocking_refreshes_then_notifies(self):
        self.send(gui.COMPLETED_STATUS)
        self.process.is_alive.return_value = False
        self.refresh.side_effect = lambda: self.assertTrue(self.state.processing)
        gui.check_upscale_steps()
        self.process.join.assert_called_once_with(timeout=0)
        self.process.kill.assert_not_called()
        self.refresh.assert_called_once()
        self.state.file_widget.clear_active_highlight.assert_called_once()
        self.allow_sleep.assert_called_once()
        self.idle_button.assert_called_once()
        self.state.info_message.set.assert_called_with("All files completed! :)")
        self.thread.assert_called_once_with(
            target=gui.show_completion_notification,
            name="QualityScaler-completion-notification", daemon=True,
        )
        self.thread.return_value.start.assert_called_once()
        self.assertFalse(gui.inputs_are_busy())
        self.assertIsNone(self.state.process_upscale_orchestrator)
        self.assertIsNone(self.state.pending_upscale_status)
        self.assertIsNone(self.state.poll_after_id)
        self.assertFalse(self.state.after_ids)
        self.send(gui.COMPLETED_STATUS)
        gui._finish_upscale(gui.COMPLETED_STATUS)
        self.thread.assert_called_once()
        self.refresh.assert_called_once()

    def test_nonzero_exit_after_completed_is_error_not_success(self):
        self.send(gui.COMPLETED_STATUS)
        self.process.is_alive.return_value = False
        self.process.exitcode = 7
        gui.check_upscale_steps()
        self.error.assert_called_once_with("Worker exited with code 7 after completion")
        self.thread.assert_not_called()
        self.process.join.assert_called_once_with(timeout=0)
        self.assertFalse(gui.inputs_are_busy())

    def test_dead_worker_without_terminal_status_never_reports_success(self):
        for exitcode in (0, 9):
            with self.subTest(exitcode=exitcode):
                self.state.processing = True
                self.state.process_upscale_orchestrator = self.process
                self.process.is_alive.return_value = False
                self.process.exitcode = exitcode
                self.error.reset_mock()
                gui.check_upscale_steps()
                self.error.assert_called_once_with(
                    "Worker exited without a completion status"
                )
                self.assertFalse(gui.inputs_are_busy())
        self.thread.assert_not_called()

    def test_exit_with_queued_progress_is_detected_on_next_poll(self):
        self.process.is_alive.return_value = False
        self.send("1. Upscaling image")
        self.assertTrue(self.state.processing)
        self.assertIsNotNone(self.state.poll_after_id)
        gui.check_upscale_steps()
        self.assertFalse(self.state.processing)
        self.error.assert_called_once_with("Worker exited without a completion status")
        self.thread.assert_not_called()

    def test_liveness_observed_before_queue_read_avoids_completion_exit_race(self):
        def observe_exit():
            self.state.process_status_q.put(gui.COMPLETED_STATUS)
            return False

        self.process.is_alive.side_effect = observe_exit
        gui.check_upscale_steps()
        self.assertFalse(self.state.processing)
        self.error.assert_not_called()
        self.thread.assert_called_once()

    def test_duplicate_terminal_and_late_progress_keep_finalizing_and_error_wins(self):
        self.send(gui.COMPLETED_STATUS)
        self.send(gui.COMPLETED_STATUS)
        self.state.info_message.set.reset_mock()
        self.send("1. Upscaling image")
        self.state.info_message.set.assert_not_called()
        self.state.file_widget.highlight_active_file.assert_not_called()
        self.send("Error cleanup failed")
        self.send(gui.COMPLETED_STATUS)
        self.process.is_alive.return_value = False
        gui.check_upscale_steps()
        self.error.assert_called_once_with("cleanup failed")
        self.thread.assert_not_called()

    def test_closeapp_queue_marker_does_not_abandon_lifecycle_polling(self):
        self.send(gui.CLOSE_APP_STATUS)
        self.assertIsNotNone(self.state.poll_after_id)
        self.process.join.assert_not_called()

    def test_stop_requests_checkpoint_without_blocking_and_waits_for_exit(self):
        with patch.object(gui, "monotonic", return_value=100):
            gui.stop_button_command()
        self.assertTrue(self.state.event_stop_upscale_process.is_set())
        self.assertEqual(self.state.stop_deadline, 110)
        self.assertTrue(gui.inputs_are_busy())
        self.process.join.assert_not_called()
        self.process.kill.assert_not_called()
        self.process.is_alive.return_value = False
        self.send(gui.COMPLETED_STATUS)  # Completion racing a user stop is not success.
        self.state.info_message.set.assert_called_with("Upscaling stopped")
        self.process.join.assert_called_once_with(timeout=0)
        self.assertFalse(self.state.event_stop_upscale_process.is_set())
        self.assertIsNone(self.state.stop_deadline)
        self.assertFalse(gui.inputs_are_busy())
        self.thread.assert_not_called()

    def test_stop_deadline_allows_checkpoint_then_kills_descendants_without_joining(self):
        child = Mock()
        with patch.object(gui, "monotonic", return_value=100) as clock, \
                patch.object(gui, "psutil_Process") as lookup:
            lookup.return_value.children.return_value = [child]
            gui.stop_button_command()
            clock.return_value = 109
            gui.stop_button_command()  # Duplicate requests cannot extend the deadline.
            self.assertEqual(self.state.stop_deadline, 110)
            lookup.assert_not_called()
            clock.return_value = 110
            gui.check_upscale_steps()
            lookup.return_value.children.assert_called_once_with(recursive=True)
            child.kill.assert_called_once()
            self.process.kill.assert_called_once()
            self.process.join.assert_not_called()
            self.assertTrue(self.state.processing)
            gui.stop_button_command()
            self.process.kill.assert_called_once()
        self.process.is_alive.return_value = False
        self.process.exitcode = -9
        gui.check_upscale_steps()
        self.process.join.assert_called_once_with(timeout=0)
        self.state.info_message.set.assert_called_with("Upscaling stopped")
        self.thread.assert_not_called()

    def test_terminal_stop_and_error_wait_for_cleanup_without_forcing_stop(self):
        for status in (gui.STOP_STATUS, "Error failed"):
            with self.subTest(status=status):
                self.state.pending_upscale_status = None
                self.send(status)
                self.assertTrue(self.state.processing)
                self.assertFalse(self.state.event_stop_upscale_process.is_set())
                self.assertIsNone(self.state.stop_deadline)
        self.process.join.assert_not_called()
        self.process.kill.assert_not_called()
        self.thread.assert_not_called()

    def test_close_keeps_polling_until_exit_then_destroys_without_toast(self):
        widget = self.state.file_widget
        with patch.object(gui, "save_user_choices_in_json") as save:
            gui.on_app_close()
            gui.on_app_close()
            save.assert_called_once()
        self.assertTrue(self.state.closing)
        self.assertTrue(self.state.processing)
        self.assertIs(self.state.window, self.window)
        self.assertTrue(self.state.event_stop_upscale_process.is_set())
        self.assertIsNone(gui._schedule(0, Mock()))
        self.process.join.assert_not_called()
        self.window.after(1, lambda: setattr(self.process.is_alive, "return_value", False))
        pump_until(self.window, lambda: self.state.window is None)
        self.process.join.assert_called_once_with(timeout=0)
        widget.destroy.assert_called_once()
        self.assertFalse(self.state.processing)
        self.assertFalse(self.state.after_ids)
        self.refresh.assert_not_called()
        self.thread.assert_not_called()

    def test_idle_close_still_destroys_when_preference_save_fails(self):
        self.state.processing = False
        self.state.process_upscale_orchestrator = None
        with patch.object(gui, "save_user_choices_in_json", side_effect=OSError("readonly")), \
                patch("builtins.print"):
            gui.on_app_close()
        self.assertTrue(self.state.closing)
        self.assertIsNone(self.state.window)
        self.thread.assert_not_called()

    def test_start_failure_restores_idle_without_joining_unstarted_process(self):
        self.state.processing = False
        self.state.process_upscale_orchestrator = None
        process = Mock(pid=None)
        process.start.side_effect = RuntimeError("spawn failed")
        with patch.object(gui, "build_processing_config", return_value=Mock()), \
                patch.object(gui, "Process", return_value=process), \
                patch.object(gui, "prevent_sleep"), \
                patch.object(gui.App, "place_stop_button"):
            gui.upscale_button_command()
        self.error.assert_called_once_with("spawn failed")
        process.join.assert_not_called()
        self.idle_button.assert_called_once()
        self.allow_sleep.assert_called_once()
        self.assertFalse(gui.inputs_are_busy())


class RealCompletionTests(unittest.TestCase):
    def test_tcl_callbacks_continue_during_worker_cleanup_and_blocked_notification(self):
        window = Tcl()
        context = get_context("spawn")
        toast_started, release_toast = Event(), Event()
        notification_threads = []
        notification_observations = []
        heartbeats = []
        main_thread = get_ident()
        with context.Manager() as manager:
            status_queue = manager.Queue(maxsize=1)
            ready, release_worker = context.Event(), context.Event()
            process = context.Process(
                target=terminal_then_wait, args=(status_queue, ready, release_worker)
            )
            state = gui.AppState(
                preferences=gui.UserPreferences(), window=window, info_message=Mock(),
                file_widget=Mock(), process_upscale_orchestrator=process,
                process_status_q=status_queue, event_stop_upscale_process=context.Event(),
                processing=True,
            )

            def heartbeat():
                heartbeats.append(monotonic())
                if state.processing or not release_toast.is_set():
                    window.after(5, heartbeat)

            def notify():
                notification_threads.append(current_thread())
                notification_observations.append((get_ident(), process.is_alive()))
                toast_started.set()
                release_toast.wait(10)

            def refresh():
                self.assertFalse(process.is_alive(), "Cards refreshed before worker exit")

            process.start()
            try:
                self.assertTrue(ready.wait(10), "Worker did not send terminal status")
                with patch.object(gui, "app_state", state), \
                        patch.object(gui, "update_file_widget", side_effect=refresh) as cards, \
                        patch.object(gui.App, "place_upscale_button"), \
                        patch.object(gui, "allow_sleep"), \
                        patch.object(gui, "show_completion_notification", side_effect=notify), \
                        patch.object(gui, "show_error_message") as error:
                    gui.check_upscale_steps()
                    window.after(0, heartbeat)
                    pump_until(window, lambda: len(heartbeats) >= 8)
                    self.assertTrue(process.is_alive())
                    self.assertTrue(gui.inputs_are_busy())
                    self.assertEqual(state.pending_upscale_status, gui.COMPLETED_STATUS)
                    self.assertFalse(state.event_stop_upscale_process.is_set())
                    cards.assert_not_called()
                    self.assertFalse(toast_started.is_set())
                    self.assertNotIn(
                        call("All files completed! :)"),
                        state.info_message.set.call_args_list,
                    )
                    release_worker.set()
                    pump_until(window, lambda: not state.processing)
                    pump_until(window, toast_started.is_set)
                    before = len(heartbeats)
                    pump_until(window, lambda: len(heartbeats) >= before + 8)
                    self.assertFalse(release_toast.is_set())
                    self.assertNotEqual(notification_observations[0][0], main_thread)
                    self.assertFalse(notification_observations[0][1])
                    self.assertTrue(notification_threads[0].daemon)
                    self.assertEqual(process.exitcode, 0)
                    cards.assert_called_once()
                    error.assert_not_called()
                    state.info_message.set.assert_called_with("All files completed! :)")
                    self.assertIsNone(state.poll_after_id)
            finally:
                release_toast.set()
                release_worker.set()
                for thread in notification_threads:
                    thread.join(timeout=2)
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                for handle in window.tk.call("after", "info"):
                    window.after_cancel(handle)


if __name__ == "__main__":
    unittest.main()
