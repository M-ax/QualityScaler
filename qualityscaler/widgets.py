"""Reusable desktop widgets with geometry owned by their parent."""

from customtkinter import CTkFrame, CTkLabel, CTkProgressBar, CTkTextbox

from .status import VideoProgress


class StatusBar(CTkFrame):
    """Wrap general status and show both concurrent video stages together."""

    def __init__(self, master, *, textvariable, font, text_color, **kwargs):
        super().__init__(master, corner_radius=0, **kwargs)
        self._variable = textvariable
        self._layout_after_id = None
        self._height = 24
        self._video_progress = None
        self.grid_columnconfigure(0, weight=1)
        self.label = CTkTextbox(
            self, font=font, text_color=text_color, fg_color="transparent",
            width=1, height=24, wrap="word", border_width=0, corner_radius=0,
            state="disabled",
        )
        self.label.grid(row=0, column=0, sticky="ew", padx=12, pady=6)
        self.video_rows = CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.video_rows.grid_columnconfigure(1, weight=1)
        self.extraction_label = CTkLabel(
            self.video_rows, text="", font=font, text_color=text_color,
            height=20, anchor="w",
        )
        self.upscaling_label = CTkLabel(
            self.video_rows, text="", font=font, text_color=text_color,
            height=20, anchor="w",
        )
        self.extraction_bar = CTkProgressBar(self.video_rows, width=1, height=8)
        self.upscaling_bar = CTkProgressBar(self.video_rows, width=1, height=8)
        for row, (label, bar) in enumerate((
            (self.extraction_label, self.extraction_bar),
            (self.upscaling_label, self.upscaling_bar),
        )):
            label.grid(row=row, column=0, sticky="w", padx=(0, 12))
            bar.grid(row=row, column=1, sticky="ew")
        self.video_rows.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        self.set_video_progress(None)
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

    def set_video_progress(self, progress: VideoProgress | None, *, status: str | None = None):
        """Apply one complete snapshot, never accumulating counts across files.

        Called only by the GUI's Tk callbacks. Unknown totals use an empty,
        static track and explicit text, not a made-up fraction or animation.
        """
        self._video_progress = progress
        if progress is None:
            self.video_rows.grid_remove()
            self.extraction_label.configure(text="")
            self.upscaling_label.configure(text="")
            self.extraction_bar.set(0)
            self.upscaling_bar.set(0)
        else:
            extraction_status = "complete" if progress.extraction_complete else "extracting"
            if progress.upscaled < progress.extracted:
                upscaling_status = "upscaling"
            elif progress.extraction_complete:
                upscaling_status = "complete"
            else:
                upscaling_status = "waiting for frames"
            for label, bar, stage, count, stage_status in (
                (self.extraction_label, self.extraction_bar, "Extraction",
                 progress.extracted, extraction_status),
                (self.upscaling_label, self.upscaling_bar, "AI upscaling",
                 progress.upscaled, upscaling_status),
            ):
                if progress.total is None:
                    detail = f"{count} frames, total unknown"
                    fraction = 0
                else:
                    estimate = "" if progress.extraction_complete else "~"
                    detail = f"{count} / {estimate}{progress.total} frames"
                    # Estimates can lag extraction; never let a bar overflow.
                    fraction = (min(1, max(0, count / progress.total))
                                if progress.total > 0 else 0)
                if status is not None and stage_status != "complete":
                    stage_status = status
                label.configure(text=f"{stage}: {detail} — {stage_status}")
                bar.set(fraction)
            self.video_rows.grid()
        self._queue_fit()

    def set_video_status(self, status: str):
        """Keep the last counters visible during cleanup and after its result."""
        if self._video_progress is not None:
            self.set_video_progress(self._video_progress, status=status)

    def _queue_fit(self, _event=None):
        if self._layout_after_id is None:
            self._layout_after_id = self.after_idle(self._fit_message)

    def _fit_message(self):
        self._layout_after_id = None
        # Tk counts real wrapped display lines at the current physical width
        # and font/DPI scale. Leave room for both video rows by capping their
        # general message at two lines (four otherwise). Text retains the entire
        # message with selection, mouse-wheel and scrollbar scrolling.
        breaks = self.label._textbox.count("1.0", "end-1c", "displaylines")
        limit = 2 if self._video_progress is not None else 4
        lines = min(limit, 1 + (breaks[0] if breaks else 0))
        height = max(24, lines * 18 + 6)
        if height != self._height:
            self._height = height
            self.label.configure(height=height)

    def destroy(self):
        self._variable.trace_remove("write", self._trace)
        if self._layout_after_id is not None:
            self.after_cancel(self._layout_after_id)
        super().destroy()
