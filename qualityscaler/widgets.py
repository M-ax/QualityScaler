"""Reusable desktop widgets with geometry owned by their parent."""

from customtkinter import CTkFrame, CTkLabel, CTkProgressBar, CTkTextbox

from .status import VideoProgress, format_duration


STAGE_COMPLETE_COLOR = "#4ADE80"


class StatusBar(CTkFrame):
    """Keep extraction, upscaling and encoding visible through job completion."""

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
        self.encoding_label = CTkLabel(
            self.video_rows, text="", font=font, text_color=text_color,
            height=20, anchor="w",
        )
        self.extraction_bar = CTkProgressBar(self.video_rows, width=1, height=8)
        self.upscaling_bar = CTkProgressBar(self.video_rows, width=1, height=8)
        self.encoding_bar = CTkProgressBar(self.video_rows, width=1, height=8)
        for row, name in enumerate(("extraction", "upscaling", "encoding")):
            check = CTkLabel(
                self.video_rows, text="", font=font, text_color=STAGE_COMPLETE_COLOR,
                width=20, height=20,
            )
            detail = CTkLabel(
                self.video_rows, text="", font=font, text_color=text_color,
                width=1, height=16, anchor="w", justify="left", wraplength=600,
            )
            setattr(self, f"{name}_check", check)
            setattr(self, f"{name}_detail", detail)
            getattr(self, f"{name}_label").grid(
                row=row * 2, column=0, sticky="w", padx=(0, 12),
            )
            getattr(self, f"{name}_bar").grid(row=row * 2, column=1, sticky="ew")
            check.grid(row=row * 2, column=2, sticky="e", padx=(6, 0))
            detail.grid(row=row * 2 + 1, column=0, columnspan=3, sticky="ew", pady=(0, 3))
        self.video_rows.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        self.video_rows.bind("<Configure>", self._queue_fit, add="+")
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
            for name in ("extraction", "upscaling", "encoding"):
                getattr(self, f"{name}_label").configure(text="")
                getattr(self, f"{name}_bar").set(0)
                getattr(self, f"{name}_check").configure(text="")
                detail = getattr(self, f"{name}_detail")
                detail.configure(text="")
                detail.grid_remove()
        else:
            stages = {stage.stage: stage for stage in getattr(progress, "stages", ())}
            extraction_status = "complete" if progress.extraction_complete else "extracting"
            if progress.upscaled < progress.extracted:
                upscaling_status = "upscaling"
            elif progress.extraction_complete:
                upscaling_status = "finalizing"
            else:
                upscaling_status = "waiting for frames"
            for key, name, title, count, stage_status in (
                ("extracting", "extraction", "Extraction", progress.extracted, extraction_status),
                ("upscaling", "upscaling", "AI upscaling", progress.upscaled, upscaling_status),
                ("encoding", "encoding", "Encoding", 0, "waiting"),
            ):
                task = stages.get(key)
                total = progress.total
                estimated = not progress.extraction_complete
                if task is not None:
                    count, total, estimated = task.completed, task.total, task.estimated_total
                    stage_status = {
                        "waiting": "waiting", "running": "encoding" if key == "encoding" else (
                            "extracting" if key == "extracting" else "upscaling"
                        ),
                        "complete": "complete", "stopped": "stopped", "error": "error",
                    }[task.state]
                # The terminal success is authoritative even when the GUI's
                # single-slot queue skipped the final completed-stage snapshot.
                if status == "job finished":
                    stage_status = "complete"
                    if total is None:
                        total = progress.total
                    if total is not None:
                        count = total
                        estimated = False
                elif status is not None and stage_status not in ("complete", "waiting"):
                    stage_status = status
                if total is None:
                    detail = f"{count} frames, total unknown"
                    fraction = 0
                else:
                    estimate = "~" if estimated else ""
                    detail = f"{count} / {estimate}{total} frames"
                    # Estimates can lag extraction; never let a bar overflow.
                    fraction = min(1, max(0, count / total)) if total > 0 else 0
                complete = (
                    (task is not None and task.state == "complete")
                    or (task is None and key == "extracting" and progress.extraction_complete)
                    or status == "job finished"
                )
                getattr(self, f"{name}_label").configure(text=f"{title}: {detail} — {stage_status}")
                getattr(self, f"{name}_bar").set(1 if complete else fraction)
                getattr(self, f"{name}_check").configure(text="✓" if complete else "")
                text = task.message if task is not None else ""
                if task is not None and stage_status == "complete":
                    if task.state == "complete" and task.eta:
                        text = f"{text} | Finished (local): {task.eta}"
                    elif status == "job finished":
                        text = "Completed successfully"
                elif task is not None and task.state == "running" and status is None:
                    remaining = (
                        format_duration(task.remaining_seconds)
                        if task.remaining_seconds is not None else "estimating..."
                    )
                    text = f"{text} | Remaining: {remaining} | ETA (local): {task.eta or 'estimating...'}"
                detail_label = getattr(self, f"{name}_detail")
                detail_label.configure(text=text)
                if text:
                    detail_label.grid()
                else:
                    detail_label.grid_remove()
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
        if self._video_progress is not None:
            # CustomTkinter applies monitor DPI once. wraplength is expressed
            # in its logical pixels, while winfo_width returns physical pixels.
            width = max(100, self._reverse_widget_scaling(self.video_rows.winfo_width()) - 12)
            for name in ("extraction", "upscaling", "encoding"):
                detail = getattr(self, f"{name}_detail")
                if detail.cget("wraplength") != width:
                    detail.configure(wraplength=width)
        # Tk counts real wrapped display lines at the current physical width
        # and font/DPI scale. Leave room for the video rows by capping their
        # general message at two lines (four otherwise). Each concurrent task
        # has its own line with timing, so give multi-task snapshots four
        # wrapped lines to keep both estimates visible at the minimum width.
        # Text retains longer messages with selection and scrolling.
        breaks = self.label._textbox.count("1.0", "end-1c", "displaylines")
        multiple_tasks = self._video_progress is not None and "\n" in self._video_progress.message
        limit = 2 if self._video_progress is not None and not multiple_tasks else 4
        lines = min(limit, 1 + (breaks[0] if breaks else 0))
        # Leave padding for the text viewport and fractional DPI rounding so
        # the last task's ETA is not clipped by a pixel at small app zooms.
        height = max(24, lines * 18 + (12 if multiple_tasks else 6))
        if height != self._height:
            self._height = height
            self.label.configure(height=height)

    def destroy(self):
        self._variable.trace_remove("write", self._trace)
        if self._layout_after_id is not None:
            self.after_cancel(self._layout_after_id)
        super().destroy()
