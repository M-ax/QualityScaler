import gc
from tkinter import Tcl
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tkinterdnd2 import COPY, DND_FILES, REFUSE_DROP

from qualityscaler.drag_drop import FileDropTarget


class FileDropTests(unittest.TestCase):
    def setUp(self):
        self.interpreter = Tcl()
        self.widget = Mock()
        self.widget.tk = self.interpreter.tk
        self.on_files = Mock(return_value=True)
        self.on_invalid = Mock()
        self.accepts = Mock(return_value=True)
        self.target = FileDropTarget(
            self.widget, self.on_files, accepts=self.accepts, on_invalid=self.on_invalid,
        )

    def tearDown(self):
        # Mock bind histories retain bound callbacks in cycles. Release the Tcl
        # interpreter here, on its owner thread, not during later worker GC.
        self.target = None
        self.widget = None
        self.interpreter = None
        gc.collect()

    def event(self, *paths):
        data = self.interpreter.tk.call("format", "%s",
                                        self.interpreter.tk.call("list", *paths))
        return SimpleNamespace(data=data, action="move")

    def test_registers_native_file_type_and_drop_events(self):
        self.widget.drop_target_register.assert_called_once_with(DND_FILES)
        self.assertEqual([call.args[0] for call in self.widget.dnd_bind.call_args_list],
                         ["<<DropEnter>>", "<<DropPosition>>", "<<Drop>>"])

    def test_tcl_list_parsing_preserves_all_filename_characters(self):
        paths = (r"C:\a folder\résumé {one}.png", r"C:\another\it's a clip.MP4")
        self.assertEqual(self.target.drop(self.event(*paths)), COPY)
        self.on_files.assert_called_once_with(paths)

    def test_drop_always_copies_and_never_moves_original_files(self):
        event = self.event(r"C:\input.png")
        self.assertEqual(event.action, "move")
        self.assertEqual(self.target.action(event), COPY)
        self.assertEqual(self.target.drop(event), COPY)

    def test_busy_or_outside_input_area_refuses_without_processing(self):
        self.accepts.return_value = False
        event = self.event(r"C:\input.png")
        self.assertEqual(self.target.action(event), REFUSE_DROP)
        self.assertEqual(self.target.drop(event), REFUSE_DROP)
        self.on_files.assert_not_called()

    def test_invalid_tcl_payload_is_refused_without_raising_in_callback(self):
        self.assertEqual(self.target.drop(SimpleNamespace(data="{unclosed")), REFUSE_DROP)
        self.on_invalid.assert_called_once()
        self.on_files.assert_not_called()

    def test_unsupported_selection_and_empty_drop_are_refused(self):
        self.on_files.return_value = False
        self.assertEqual(self.target.drop(self.event(r"C:\notes.txt")), REFUSE_DROP)
        self.on_files.reset_mock()
        self.assertEqual(self.target.drop(self.event()), REFUSE_DROP)
        self.on_files.assert_not_called()


if __name__ == "__main__":
    unittest.main()
