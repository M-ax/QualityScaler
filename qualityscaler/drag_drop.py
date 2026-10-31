"""Native file drops for CustomTkinter, including the portable executable."""

from tkinter import TclError

from customtkinter import CTk
from tkinterdnd2 import COPY, DND_FILES, REFUSE_DROP, TkinterDnD


class FileDropWindow(CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.TkdndVersion = TkinterDnD._require(self)
        except Exception:
            self.destroy()
            raise


class FileDropTarget:
    """Route native drops into the same file-selection callback as the picker."""

    def __init__(self, widget, on_files, *, accepts=None, on_invalid=None):
        self.widget = widget
        self.on_files = on_files
        self.accepts = accepts if accepts is not None else lambda event: True
        self.on_invalid = on_invalid if on_invalid is not None else lambda: None
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<DropEnter>>", self.action)
        widget.dnd_bind("<<DropPosition>>", self.action)
        widget.dnd_bind("<<Drop>>", self.drop)

    def action(self, event):
        return COPY if self.accepts(event) else REFUSE_DROP

    def drop(self, event):
        if not self.accepts(event):
            return REFUSE_DROP
        try:
            # Native drops are Tcl lists, not whitespace-separated filenames.
            # splitlist preserves spaces, Unicode, braces and backslashes.
            paths = self.widget.tk.splitlist(event.data)
        except TclError:
            self.on_invalid()
            return REFUSE_DROP
        return COPY if paths and self.on_files(paths) else REFUSE_DROP
