"""Opt-in preview validation with the locally provisioned standard model bundle."""

from dataclasses import replace
import gc
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest

import cv2
import numpy as np

from qualityscaler.ai import get_model_upscale_factor
from qualityscaler.catalog import MODEL_NAMES
from qualityscaler.config import ProcessingConfig
from qualityscaler.geometry import VideoGeometry, scaled_size
from qualityscaler.media import image_write
from qualityscaler.preview import create_preview_engine, probe_source, render_preview


def preview_config(model):
    return ProcessingConfig(
        [], "unused", model, 1, 0.5, 1.0, "Auto", 512,
        0.3, False, ".png", ".mp4", "x264",
    )


@unittest.skipUnless(
    os.environ.get("QUALITYSCALER_TEST_MODEL"),
    "Set QUALITYSCALER_TEST_MODEL to validate previews with real ONNX models",
)
class RealPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.pixels = np.random.default_rng(42).integers(0, 256, (32, 48, 3), dtype=np.uint8)
        self.image = self.directory / "résumé preview.png"
        image_write(str(self.image), self.pixels, ".png")
        self.video = self.directory / "seek preview.avi"
        writer = cv2.VideoWriter(
            str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 4, (48, 32),
        )
        self.assertTrue(writer.isOpened())
        try:
            for index in range(4):
                writer.write(np.roll(self.pixels, index * 8, axis=1))
        finally:
            writer.release()

    def test_all_nine_models_render_still_and_seeked_video_with_real_inference(self):
        image_source = probe_source(str(self.image))
        video_source = probe_source(str(self.video))
        for model in MODEL_NAMES:
            with self.subTest(model=model):
                config = replace(preview_config(model), output_resize_factor=0.75)
                factor = get_model_upscale_factor(model)
                engine = create_preview_engine(config)
                try:
                    still = render_preview(image_source, 0, config, engine)
                    input_width, input_height = scaled_size(48, 32, config.input_resize_factor)
                    width, height = scaled_size(
                        input_width * factor, input_height * factor, config.output_resize_factor,
                    )
                    self.assertEqual(still.shape, (height, width, 3))
                    self.assertEqual(still.dtype, np.uint8)
                    self.assertGreater(int(still.max()), int(still.min()))
                    frame = render_preview(video_source, 3, config, engine)
                    geometry = VideoGeometry.calculate(
                        48, 32, config.input_resize_factor, factor, config.output_resize_factor,
                    )
                    self.assertEqual(frame.shape, (*geometry.output_size[::-1], 3))
                    self.assertEqual(frame.dtype, np.uint8)
                    self.assertGreater(int(frame.max()), int(frame.min()))
                    print(f"Real preview {model}: still {still.shape}, seek frame 3 {frame.shape}")
                finally:
                    del engine
                    gc.collect()

    @unittest.skipUnless(sys.platform == "win32" or os.environ.get("DISPLAY"),
                         "Real preview window needs a desktop display")
    def test_real_window_worker_seeks_refreshes_zooms_and_releases_session(self):
        from customtkinter import CTk
        from qualityscaler.preview_window import UpscalePreview

        config = preview_config(os.environ["QUALITYSCALER_TEST_MODEL"])
        root = CTk()
        window = None

        def wait_for_image():
            deadline = monotonic() + 90
            while window.viewport.image is None and monotonic() < deadline:
                root.update()
                self.assertFalse(window._failed, window.status.cget("text"))
                self.assertNotIn("error", window.status.cget("text").lower())
                sleep(0.01)
            self.assertIsNotNone(window.viewport.image, window.status.cget("text"))
            root.update()

        try:
            window = UpscalePreview(
                root, [str(self.video), str(self.image)], lambda: config, lambda: None,
            )
            process = window._worker.process
            reader = window._worker._reader
            self.assertNotEqual(process.pid, os.getpid())
            wait_for_image()
            first = np.asarray(window.viewport.image).copy()
            window.seek.set(3)
            window._seek(window.seek.get())
            wait_for_image()
            self.assertEqual(window._index, 3)
            self.assertFalse(np.array_equal(np.asarray(window.viewport.image), first))
            self.assertIn("Frame 4 / 4", window.position.cget("text"))
            config = replace(config, output_resize_factor=0.75, selected_sharpening_amount=0.5)
            window.settings_changed()
            wait_for_image()
            geometry = VideoGeometry.calculate(
                48, 32, config.input_resize_factor,
                get_model_upscale_factor(config.selected_AI_model), config.output_resize_factor,
            )
            self.assertEqual(window.viewport.image.size, geometry.output_size)
            scale = window.viewport.view.scale
            window.viewport.canvas.event_generate("<MouseWheel>", delta=120, x=100, y=100)
            root.update()
            self.assertGreater(window.viewport.view.scale, scale)
            window.files.set(window._labels[1])
            window._select_file(window.files.get())
            wait_for_image()
            self.assertEqual(window.seek.cget("state"), "disabled")
            self.assertEqual(window.position.cget("text"), "Still image")
            window.destroy()
            self.assertFalse(process.is_alive())
            self.assertFalse(reader.is_alive())
        finally:
            if window is not None:
                window.destroy()
            for handle in root.tk.splitlist(root.tk.call("after", "info")):
                root.tk.call("after", "cancel", handle)
            root.destroy()
            gc.collect()


if __name__ == "__main__":
    unittest.main()
