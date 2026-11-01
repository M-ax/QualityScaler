"""Reusable desktop widgets with geometry owned by their parent."""

from customtkinter import CTkFrame, CTkTextbox


class StatusBar(CTkFrame):
    """Wrap status text, scrolling excess without hiding the job controls."""

    def __init__(self, master, *, textvariable, font, text_color, **kwargs):
        super().__init__(master, corner_radius=0, **kwargs)
        self._variable = textvariable
        self._layout_after_id = None
        self._height = 24
        self.grid_columnconfigure(0, weight=1)
        self.label = CTkTextbox(
            self, font=font, text_color=text_color, fg_color="transparent",
            width=1, height=24, wrap="word", border_width=0, corner_radius=0,
            state="disabled",
        )
        self.label.grid(row=0, column=0, sticky="ew", padx=12, pady=6)
        self._trace = textvariable.trace_add("write", self._replace_message)
        self.bind("<Configure>", self._queue_fit, add="+")
        # The text viewport is laid out after the outer frame, especially when
        # the root starts withdrawn. Recount once its actual width is known.
        self.label.bind("<Configure>", self._queue_fit, add="+")
        self._replace_message()

    def _replace_message(self, *_args):
        self.label.configure(state="normal")
        self.label.delete("1.0", "end")
        self.label.insert("1.0", self._variable.get())
        self.label.configure(state="disabled")
        self.label.yview_moveto(0)
        self._queue_fit()

    def _queue_fit(self, _event=None):
        if self._layout_after_id is None:
            self._layout_after_id = self.after_idle(self._fit_message)

    def _fit_message(self):
        self._layout_after_id = None
        # Tk counts real wrapped display lines at the current physical width
        # and font/DPI scale. Cap the bar at four lines; Text retains the entire
        # message and supports selection, mouse-wheel and scrollbar scrolling.
        breaks = self.label._textbox.count("1.0", "end-1c", "displaylines")
        lines = min(4, 1 + (breaks[0] if breaks else 0))
        height = max(24, lines * 18 + 6)
        if height != self._height:
            self._height = height
            self.label.configure(height=height)

    def destroy(self):
        self._variable.trace_remove("write", self._trace)
        if self._layout_after_id is not None:
            self.after_cancel(self._layout_after_id)
        super().destroy()
