"""Preview scheduling, isolation, real Tk controls, and viewport-only zoom."""

from dataclasses import replace
import os
from pathlib import Path
import pickle
from queue import Queue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from qualityscaler import gui, preview_window, preview_worker
from qualityscaler.config import ProcessingConfig
from qualityscaler.preview_window import ImageViewport, UpscalePreview, ZoomView
from qualityscaler.preview_worker import PreviewRequest, PreviewResult


def config():
    return ProcessingConfig(
        ["clip.mp4"], "unused", "LVAx2", 1, 0.5, 1.0,
        "Auto", 800, 0.3, True, ".png", ".mp4", "x264",
    )


class ZoomTests(unittest.TestCase):
    def test_render_converts_only_viewport_pixels_not_entire_upscale(self):
        viewport = object.__new__(ImageViewport)
        viewport.image = Image.new("RGBA", (1024, 768)).convert("RGBa")
        viewport.view = ZoomView(1024, 768, 2)
        viewport.canvas = Mock()
        viewport.canvas.winfo_width.return_value = 80
        viewport.canvas.winfo_height.return_value = 60
        viewport._image_item = 1
        viewport._on_zoom = Mock()
        viewport._on_error = Mock()
        converted_sizes = []
        convert = Image.Image.convert

        def track_convert(image, *args, **kwargs):
            converted_sizes.append(image.size)
            return convert(image, *args, **kwargs)

        with patch.object(Image.Image, "convert", new=track_convert), \
                patch.object(preview_window.ImageTk, "PhotoImage") as photo:
            viewport._render_view()
        self.assertEqual(converted_sizes, [(80, 60)])
        self.assertEqual(photo.call_args.args[0].mode, "RGBA")
        viewport._on_error.assert_not_called()

    def test_fit_preserves_aspect_and_centers_without_upscaling(self):
        view = ZoomView(1920, 1080)
        view.fit(800, 600)
        self.assertAlmostEqual(view.scale, 800 / 1920)
        self.assertEqual(view.x, 0)
        self.assertEqual(view.y, 75)
        view = ZoomView(40, 20)
        view.fit(800, 600)
        self.assertEqual(view.scale, 1)
        self.assertEqual((view.x, view.y), (380, 290))

    def test_zoom_keeps_pointer_over_same_source_pixel(self):
        view = ZoomView(2000, 1000, 1, -300, -100)
        before = ((250 - view.x) / view.scale, (200 - view.y) / view.scale)
        view.zoom_at(1.2, 250, 200, 800, 600)
        after = ((250 - view.x) / view.scale, (200 - view.y) / view.scale)
        self.assertEqual(before, after)

    def test_zoom_and_pan_are_bounded_and_fit_recenters(self):
        view = ZoomView(2000, 1000, 1, 900, -99999)
        view.constrain(800, 600)
        self.assertEqual((view.x, view.y), (0, -400))
        view.zoom_at(1e9, 400, 300, 800, 600)
        self.assertEqual(view.scale, 16)
        view.zoom_at(1e-12, 400, 300, 800, 600)
        self.assertAlmostEqual(view.scale, 0.4 / 8)
        self.assertGreater(view.x, 0)
        self.assertGreater(view.y, 0)
        view.fit(800, 600)
        self.assertEqual((view.x, view.y), (0, 100))


class PreviewSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.window = object.__new__(UpscalePreview)
        window = self.window
        window._closed = window._failed = False
        window._generation = 0
        window._pending = None
        window._inflight = False
        window._seek_id = window._poll_id = None
        window._settings_key = None
        window._path = "clip.mp4"
        window._index = 0
        window._source = SimpleNamespace(
            path="clip.mp4", frame_count=100, fps=25, is_video=True,
        )
        window._get_config = Mock(return_value=config())
        window._worker = Mock()
        window.status = Mock()
        window.viewport = Mock()
        window.seek = Mock()
        window.previous = Mock()
        window.next = Mock()
        window.position = Mock()
        window.after = Mock(side_effect=lambda *_args: "after#1")
        window.after_cancel = Mock()

    def test_seeks_coalesce_and_only_latest_pending_request_is_submitted(self):
        window = self.window
        window._seek(10)
        window._dispatch()
        first = window._worker.submit.call_args.args[0]
        window._seek(20)
        window._seek(90)
        window._dispatch()
        window._worker.submit.assert_called_once()
        self.assertEqual(window._pending.frame_index, 90)
        window._worker.receive.return_value = PreviewResult(
            first.generation, window._source, first.frame_index, image=Mock(),
        )
        window._poll()
        window.viewport.set_image.assert_not_called()
        self.assertEqual(window._worker.submit.call_count, 2)
        latest = window._worker.submit.call_args.args[0]
        self.assertEqual(latest.frame_index, 90)
        self.assertEqual(latest.generation, window._generation)

    def test_invalid_settings_invalidate_result_and_show_error(self):
        window = self.window
        window.refresh()
        generation = window._generation
        window._get_config.side_effect = ValueError("Input resolution % must be a number")
        window.refresh()
        self.assertGreater(window._generation, generation)
        self.assertIsNone(window._pending)
        window.status.configure.assert_called_with(text="Input resolution % must be a number")
        window.viewport.clear.assert_called()
        window._get_config.side_effect = None
        window.settings_changed()
        self.assertIsNotNone(window._pending)

    def test_nonpixel_changes_do_not_repeat_inference(self):
        window = self.window
        window._settings_key = window._config_key(config())
        window._get_config.return_value = replace(
            config(), selected_keep_frames=False, selected_video_codec="x265",
            selected_output_path="elsewhere",
        )
        window.settings_changed()
        self.assertIsNone(window._pending)
        window._get_config.return_value.output_resize_factor = 0.75
        window.settings_changed()
        self.assertIsNotNone(window._pending)

    def test_worker_exit_cancels_pending_timer_and_does_not_poll_forever(self):
        window = self.window
        window._seek_id = "seek#1"
        window._worker.receive.side_effect = RuntimeError("worker exited")
        window._poll()
        self.assertTrue(window._failed)
        window.after_cancel.assert_called_once_with("seek#1")
        window.after.assert_not_called()
        self.assertIsNone(window._poll_id)
        window.refresh()
        window._select_file("another file")
        window._queue(PreviewRequest(2, "another.mp4"))
        window._dispatch()
        window._worker.submit.assert_not_called()
        self.assertIsNone(window._pending)

    def test_display_error_is_reported_without_stopping_polling_or_retry(self):
        window = self.window
        window._worker.receive.return_value = PreviewResult(
            0, window._source, image=Image.new("RGBA", (8, 6)),
        )
        window.viewport.set_image.side_effect = ValueError("display failed")
        window._poll()
        window.status.configure.assert_called_with(text="Preview display error: display failed")
        self.assertIsNotNone(window._poll_id)
        window.viewport.set_image.side_effect = None
        window._poll()
        self.assertTrue(window.status.configure.call_args.kwargs["text"].startswith("Upscaled"))

    def test_metadata_probe_enables_integer_frame_seek_then_requests_inference(self):
        window = self.window
        window._worker.receive.return_value = PreviewResult(0, window._source)
        window._poll()
        window.seek.configure.assert_any_call(to=99, number_of_steps=99)
        window.seek.configure.assert_any_call(state="normal")
        self.assertIsNotNone(window._pending.config)
        window.position.configure.assert_called_with(
            text="Frame 1 / 100 · 00:00.000 / 00:03.960",
        )


class PreviewWiringTests(unittest.TestCase):
    def setUp(self):
        self.state = gui.AppState(
            preferences=gui.UserPreferences(), window=Mock(), info_message=Mock(),
        )
        self.state.file_widget = Mock()
        self.state.file_widget.get_selected_file_list.return_value = ["clip.mp4", "photo.png"]
        patcher = patch.object(gui, "app_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_preview_uses_current_config_without_disk_space_check_and_reuses_window(self):
        with patch.object(gui, "_processing_config", return_value=config()) as build, \
                patch.object(gui, "check_disk_space") as disk, \
                patch.object(gui, "UpscalePreview") as constructor:
            gui.open_upscale_preview()
            self.assertIs(self.state.upscale_preview, constructor.return_value)
            args = constructor.call_args.args
            self.assertEqual(args[:2], (self.state.window, ["clip.mp4", "photo.png"]))
            self.assertEqual(args[2](), config())
            self.assertEqual(build.call_count, 2)
            gui.open_upscale_preview()
            constructor.assert_called_once()
            constructor.return_value.lift.assert_called_once()
            disk.assert_not_called()
            args[3]()
            self.assertIsNone(self.state.upscale_preview)

    def test_busy_or_empty_inputs_do_not_open_preview(self):
        with patch.object(gui, "UpscalePreview") as constructor:
            self.state.processing = True
            gui.open_upscale_preview()
            self.state.processing = False
            self.state.file_widget.get_selected_file_list.return_value = []
            gui.open_upscale_preview()
        constructor.assert_not_called()
        self.state.info_message.set.assert_called_with("Please select a file to preview")

    def test_start_closes_preview_before_batch_worker_is_started(self):
        preview = self.state.upscale_preview = Mock()
        self.state.process_status_q = Queue()
        self.state.event_stop_upscale_process = Mock()
        calls = []
        preview.destroy.side_effect = lambda: calls.append("preview closed")
        process = Mock()
        process.start.side_effect = lambda: calls.append("batch started")
        with patch.object(gui, "build_processing_config", return_value=config()), \
                patch.object(gui, "Process", return_value=process), \
                patch.object(gui.App, "place_stop_button"), \
                patch.object(gui, "prevent_sleep"), patch.object(gui, "_schedule"):
            gui.upscale_button_command()
        self.assertEqual(calls, ["preview closed", "batch started"])
        self.assertIsNone(self.state.upscale_preview)


class PreviewWorkerTests(unittest.TestCase):
    def test_close_terminates_hung_inference_and_closes_ipc(self):
        worker = object.__new__(preview_worker.PreviewWorker)
        worker.process = Mock(pid=123)
        worker.process.is_alive.return_value = True
        worker.requests = Mock()
        worker._closed = Event()
        worker._receiver = Mock()
        worker._reader = Mock()
        worker.close()
        worker.close()
        worker.process.terminate.assert_called_once()
        worker.process.kill.assert_called_once()
        self.assertEqual(worker.process.join.call_count, 2)
        worker._reader.join.assert_called_once_with(timeout=0.2)
        worker._receiver.close.assert_called_once()
        worker.requests.cancel_join_thread.assert_called_once()
        worker.requests.close.assert_called_once()

    def test_partial_result_transfer_never_blocks_ui_receive(self):
        worker = object.__new__(preview_worker.PreviewWorker)
        worker._completed = Queue()
        worker._closed = Event()
        started, complete = Event(), Event()
        result = PreviewResult(1, image=Image.new("RGBA", (8, 6)))

        def receive():
            if started.is_set():
                raise EOFError()
            started.set()
            if not complete.wait(timeout=5):
                raise RuntimeError("test timed out")
            return result

        worker._receiver = Mock()
        worker._receiver.recv.side_effect = receive
        reader = Thread(target=worker._receive_results)
        reader.start()
        try:
            self.assertTrue(started.wait(timeout=5))
            self.assertIsNone(worker.receive())
            complete.set()
        finally:
            complete.set()
            reader.join(timeout=5)
        self.assertFalse(reader.is_alive())
        self.assertIs(worker.receive(), result)

    def test_worker_reuses_session_for_seeks_and_postprocessing_changes(self):
        from qualityscaler import preview

        requests, results = Queue(), Queue()
        first = config()
        for index, settings in enumerate((
            first, replace(first, output_resize_factor=0.75),
            replace(first, selected_AI_model="BSRGANx2"),
        )):
            requests.put(PreviewRequest(index, "clip.mp4", 20, settings))
        requests.put(None)
        source = SimpleNamespace(path="clip.mp4", frame_count=10, fps=25, is_video=True)
        pixels = np.zeros((4, 6, 3), np.uint8)
        with patch.object(preview, "probe_source", return_value=source) as probe, \
                patch.object(preview, "create_preview_engine") as engine, \
                patch.object(preview, "render_preview", return_value=pixels) as render:
            preview_worker.run_preview_worker(requests, SimpleNamespace(send=results.put))
        probe.assert_called_once_with("clip.mp4")
        self.assertEqual(engine.call_count, 2)
        self.assertEqual(render.call_count, 3)
        for generation in range(3):
            result = results.get_nowait()
            self.assertEqual(result.generation, generation)
            self.assertEqual(result.frame_index, 9)
            self.assertEqual(result.image.mode, "RGBa")
            np.testing.assert_array_equal(np.asarray(result.image.convert("RGBA"))[:, :, :3], pixels)

    def test_worker_reports_error_and_accepts_later_retry(self):
        from qualityscaler import preview

        requests, results = Queue(), Queue()
        requests.put(PreviewRequest(1, "broken.mp4"))
        requests.put(PreviewRequest(2, "photo.png"))
        requests.put(None)
        source = SimpleNamespace(path="photo.png", frame_count=1, fps=30, is_video=False)
        with patch.object(preview, "probe_source", side_effect=[ValueError("unreadable"), source]):
            preview_worker.run_preview_worker(requests, SimpleNamespace(send=results.put))
        self.assertEqual(results.get_nowait().error, "unreadable")
        self.assertIs(results.get_nowait().source, source)

    def test_display_transport_keeps_bgr_alpha_and_scales_16bit(self):
        pixels = np.array([[[0, 25700, 65535, 65535]]], np.uint16)
        image = preview_worker._display_image(pixels)
        self.assertEqual(image.getpixel((0, 0)), (255, 100, 0, 255))

    def test_trusted_raw_transport_does_not_apply_encoded_image_bomb_limits(self):
        with patch.object(Image, "MAX_IMAGE_PIXELS", 1):
            image = preview_worker._display_image(np.zeros((8, 6, 3), np.uint8))
            restored = pickle.loads(pickle.dumps(PreviewResult(1, image=image)))
        self.assertEqual(restored.image.size, (6, 8))

    def test_spawn_worker_processes_transfers_pixels_and_exits_without_gui_imports(self):
        # Run from a real file: Windows spawn must not depend on Tk's __main__.
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            image = Path(directory) / "résumé.png"
            Image.new("RGB", (8, 6), (200, 100, 30)).save(image)
            script = Path(directory) / "probe.py"
            script.write_text("""
import builtins, sys, time
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'tkinter', 'customtkinter', 'onnxruntime'}:
        raise AssertionError('Preview probe imported GUI/runtime: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import numpy as np
from types import SimpleNamespace
from qualityscaler.ai import AI_upscale
from qualityscaler.config import ProcessingConfig
from qualityscaler.preview_worker import PreviewWorker, PreviewRequest
def fake_session(self):
    return SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name='input')],
        run=lambda _, feeds: [
            np.repeat(np.repeat(feeds['input'], 2, axis=2), 2, axis=3)
        ],
    )
AI_upscale._load_inferenceSession = fake_session
if __name__ == '__main__':
    worker = PreviewWorker()
    process = worker.process
    reader = worker._reader
    try:
        worker.submit(PreviewRequest(1, sys.argv[2]))
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = worker.receive()
            if result is not None:
                assert result.error is None, result.error
                assert result.source.frame_count == 1
                break
            time.sleep(0.02)
        else:
            raise AssertionError('Preview worker timed out')
        settings = ProcessingConfig(
            [sys.argv[2]], 'unused', 'LVAx2', 1, 1.0, 1.0,
            'Auto', 800, 0.0, False, '.png', '.mp4', 'x264',
        )
        worker.submit(PreviewRequest(2, sys.argv[2], 0, settings))
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = worker.receive()
            if result is not None:
                assert result.error is None, result.error
                assert result.image.mode == 'RGBa'
                assert result.image.size == (16, 12), result.image.size
                actual = result.image.getpixel((0, 0))
                assert all(abs(a - b) <= 1 for a, b in zip(actual, (200, 100, 30, 255))), actual
                break
            time.sleep(0.02)
        else:
            raise AssertionError('Preview render timed out')
    finally:
        worker.close()
    assert not process.is_alive()
    assert not reader.is_alive()
""", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(script), str(root), str(image)],
                capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(sys.platform == "win32" or os.environ.get("DISPLAY"),
                     "Preview smoke requires a desktop display")
class PreviewDesktopTests(unittest.TestCase):
    def test_real_window_seeks_displays_zooms_and_cleans_up(self):
        from qualityscaler.preview import PreviewSource

        class FakeWorker:
            def __init__(self):
                self.request = None
                self.closed = False
                self.submissions = []

            def submit(self, request):
                self.request = request
                self.submissions.append(request)

            def receive(self):
                request, self.request = self.request, None
                if request is None:
                    return None
                image = None
                if request.config is not None:
                    image = Image.new(
                        "RGBA", (1200, 800), (200, request.frame_index, 20, 255),
                    ).convert("RGBa")
                return PreviewResult(
                    request.generation, PreviewSource(request.path, 50, 25, True),
                    request.frame_index, image,
                )

            def close(self):
                self.closed = True

        def pump_until(root, predicate):
            deadline = monotonic() + 5
            while not predicate() and monotonic() < deadline:
                root.update()
            self.assertTrue(predicate(), "Preview UI timed out")

        root = gui.CTk()
        closed = Mock()
        worker = FakeWorker()
        window = None
        try:
            with patch.object(preview_window, "PreviewWorker", return_value=worker):
                window = UpscalePreview(root, ["clip.mp4"], config, closed)
            pump_until(root, lambda: window.viewport.image is not None)
            self.assertEqual(window.seek.cget("to"), 49)
            self.assertEqual(window.seek.cget("state"), "normal")
            window.seek.set(25)
            window._seek(window.seek.get())
            pump_until(root, lambda: window.viewport.image is not None)
            self.assertEqual(worker.submissions[-1].frame_index, 25)
            self.assertIn("Frame 26 / 50", window.position.cget("text"))
            canvas = window.viewport.canvas
            window.viewport.actual_size()
            root.update()
            before = window.viewport.view.scale
            canvas.event_generate("<MouseWheel>", delta=120, x=200, y=150)
            root.update()
            self.assertGreater(window.viewport.view.scale, before)
            submissions = len(worker.submissions)
            canvas.event_generate("<ButtonPress-1>", x=200, y=150)
            canvas.event_generate("<B1-Motion>", x=150, y=100)
            root.update()
            self.assertEqual(len(worker.submissions), submissions)
            self.assertEqual(window.viewport._photo.width(), canvas.winfo_width())
            self.assertEqual(window.viewport._photo.height(), canvas.winfo_height())
            window.destroy()
            window.destroy()  # Idempotent even after its worker and Tk widgets are gone.
            self.assertTrue(worker.closed)
            closed.assert_called_once()
            self.assertIsNone(window._poll_id)
            self.assertIsNone(window._seek_id)
        finally:
            if window is not None:
                window.destroy()
            for handle in root.tk.splitlist(root.tk.call("after", "info")):
                root.tk.call("after", "cancel", handle)
            root.destroy()


if __name__ == "__main__":
    unittest.main()
