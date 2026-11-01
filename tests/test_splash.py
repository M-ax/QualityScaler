"""Exercise the embedded splash extension with real Tcl/Tk, without PyInstaller."""

import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "qualityscaler/splash.tcl").read_text(encoding="utf-8")


@unittest.skipUnless(sys.platform == "win32" or os.environ.get("DISPLAY"),
                     "Native splash rendering requires a desktop display")
class NativeSplashRenderingTests(unittest.TestCase):
    def setUp(self):
        import tkinter

        try:
            self.window = tkinter.Tk()
        except tkinter.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.addCleanup(self.window.destroy)
        self.window.withdraw()
        self.tk = self.window.tk
        # Deliberately not divisible by the block sizes, to catch edge clipping.
        self.width, self.height = 479, 317
        self.tk.call("image", "create", "photo", "splash_image",
                     "-width", self.width, "-height", self.height)
        for x in range(self.width):
            color = f"#{x % 256:02x}{(x * 7) % 256:02x}{(x * 13) % 256:02x}"
            self.tk.call("splash_image", "put", color,
                         "-to", x, 0, x + 1, self.height)
        # These are the image/canvas/text contracts of the pinned native template.
        self.tk.eval("""
            set image_width [image width splash_image]
            set image_height [image height splash_image]
            frame .root
            canvas .root.canvas -width $image_width -height $image_height
            .root.canvas create image 0 0 -anchor nw -image splash_image
            .root.canvas create text 16 300 -tag vartext -anchor sw
            proc canvas_text_update {canvas tag _var - -} {
                upvar #0 $_var text
                $canvas itemconfigure $tag -text $text
            }
            trace add variable status_text write \
                [list canvas_text_update .root.canvas vartext]
            set status_text "Starting QualityScaler..."
            set qs_extraction_files 10
            # PyInstaller's minimal native initialization does not load these.
            rename ::tcl::mathfunc::min {}
            rename ::tcl::mathfunc::max {}
        """)
        self.tk.eval(SCRIPT)
        self.addCleanup(self.cancel_timers)

    def cancel_timers(self):
        for timer in self.tk.splitlist(self.tk.call("after", "info")):
            self.tk.call("after", "cancel", timer)

    def status(self, text):
        self.tk.setvar("status_text", text)

    def settle(self):
        # Deterministic animation frames, not a sleep or a wall-clock assertion.
        for _ in range(100):
            self.tk.call("qs_tick")
        self.cancel_timers()

    def progress(self):
        return float(self.tk.getvar("qs_progress"))

    def bar_coords(self):
        return tuple(float(value) for value in self.tk.splitlist(
            self.tk.call(".root.canvas", "coords", self.tk.getvar("qs_bar")),
        ))

    def test_first_frame_is_pixelated_and_bar_is_flush_with_bottom_edge(self):
        self.assertEqual(self.progress(), 0)
        self.assertEqual(int(self.tk.getvar("qs_last_block")), 24)
        self.assertEqual(self.bar_coords(), (0, self.height - 4, 0, self.height))
        self.assertEqual(self.tk.call("qs_frame", "get", 0, 0),
                         self.tk.call("qs_frame", "get", 23, 0))
        self.assertNotEqual(self.tk.call("splash_image", "get", 0, 0),
                            self.tk.call("splash_image", "get", 23, 0))
        self.assertEqual(int(self.tk.call("image", "width", "qs_frame")), self.width)
        self.assertEqual(int(self.tk.call("image", "height", "qs_frame")), self.height)
        # The right/bottom edge is filled, not transparent after zoom/cropping.
        self.assertEqual(int(self.tk.call("qs_frame", "transparency", "get",
                                         self.width - 1, self.height - 1)), 0)

    def test_extraction_progress_is_eased_bounded_and_does_not_finish_on_its_own(self):
        for index in range(20):
            self.status(f"AI-onnx/model {index}.onnx")
        self.assertEqual(self.progress(), 0)
        self.tk.call("qs_tick")
        self.assertTrue(0 < self.progress() < 0.7)
        self.settle()
        self.assertAlmostEqual(self.progress(), 0.7)
        self.assertTrue(1 < int(self.tk.getvar("qs_last_block")) < 24)
        before = self.progress()
        self.settle()
        self.assertEqual(self.progress(), before)
        self.assertLess(self.bar_coords()[2], self.width)

    def test_phase_messages_show_only_status_and_cannot_move_progress_backwards(self):
        self.status("QS_PROGRESS:88:Detecting adapters: résumé {GPU} (0)")
        self.settle()
        self.assertAlmostEqual(self.progress(), 0.88)
        self.assertEqual(self.tk.call(".root.canvas", "itemcget", "vartext", "-text"),
                         "Detecting adapters: résumé {GPU} (0)")
        self.status("QS_PROGRESS:72:Earlier phase")
        self.status("A plain status is still supported")
        self.settle()
        self.assertAlmostEqual(self.progress(), 0.88)
        self.assertEqual(self.tk.call(".root.canvas", "itemcget", "vartext", "-text"),
                         "A plain status is still supported")

    def test_ready_immediately_restores_exact_original_and_fills_bar(self):
        self.status("QS_PROGRESS:100:Ready")
        self.assertEqual(self.progress(), 1)
        self.assertEqual(int(self.tk.getvar("qs_last_block")), 1)
        self.assertEqual(self.bar_coords(),
                         (0, self.height - 4, self.width, self.height))
        self.assertEqual(self.tk.call("qs_frame", "data", "-format", "ppm"),
                         self.tk.call("splash_image", "data", "-format", "ppm"))
        self.tk.call("qs_tick")
        # Completion stops scheduling animation frames.
        self.cancel_timers()
        self.tk.call("qs_tick")
        self.assertFalse(self.tk.call("after", "info"))


if __name__ == "__main__":
    unittest.main()
