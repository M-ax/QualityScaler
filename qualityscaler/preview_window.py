"""Nonmodal upscale preview with debounced seeking and viewport-only zoom."""

from dataclasses import dataclass
from pathlib import Path
from tkinter import Canvas, TclError

from customtkinter import CTkButton, CTkFrame, CTkLabel, CTkOptionMenu, CTkSlider, CTkToplevel
from PIL import Image, ImageTk

from .preview_worker import PreviewRequest, PreviewWorker


@dataclass
class ZoomView:
    """Image-to-canvas transform in physical pixels; no zoomed full-image copies."""

    width: int = 1
    height: int = 1
    scale: float = 1.0
    x: float = 0.0
    y: float = 0.0

    def fit(self, width, height):
        self.scale = min(width / self.width, height / self.height, 1.0)
        self.constrain(width, height)

    def constrain(self, width, height):
        for position, extent, viewport in (
            ("x", self.width * self.scale, width),
            ("y", self.height * self.scale, height),
        ):
            value = ((viewport - extent) / 2 if extent <= viewport else
                     min(0, max(viewport - extent, getattr(self, position))))
            setattr(self, position, value)

    def zoom_at(self, factor, x, y, width, height):
        minimum = min(width / self.width, height / self.height, 1.0) / 8
        scale = min(16.0, max(minimum, self.scale * factor))
        ratio = scale / self.scale
        self.x = x - (x - self.x) * ratio
        self.y = y - (y - self.y) * ratio
        self.scale = scale
        self.constrain(width, height)


class ImageViewport(CTkFrame):
    """Wheel zoom centered at the pointer; left-drag pans the scaled result."""

    def __init__(self, master, on_zoom, on_error, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.canvas = Canvas(self, bg="#101010", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.view = ZoomView()
        self.image = None
        self._photo = None
        self._draw_id = None
        self._fit_mode = True
        self._drag = None
        self._on_zoom = on_zoom
        self._on_error = on_error
        self._image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Configure>", self._resize)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", self._wheel)
        self.canvas.bind("<Button-5>", self._wheel)
        self.canvas.bind("<ButtonPress-1>", self._start_pan)
        self.canvas.bind("<B1-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-1>", self._end_pan)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.fit())

    def set_image(self, image):
        changed_size = image.size != (self.view.width, self.view.height)
        self.image = image  # Already premultiplied RGBa; no full-image copy on Tk.
        self.view.width, self.view.height = image.size
        if changed_size or self._fit_mode:
            self.fit()
        else:
            self._queue_draw()

    def clear(self):
        self.image = self._photo = None
        self.canvas.itemconfigure(self._image_item, image="")

    def fit(self):
        self._fit_mode = True
        self.view.fit(self.canvas.winfo_width(), self.canvas.winfo_height())
        self._queue_draw()

    def actual_size(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self._fit_mode = False
        self.view.zoom_at(1 / self.view.scale, width / 2, height / 2, width, height)
        self._queue_draw()

    def _resize(self, _event):
        if self._fit_mode:
            self.view.fit(self.canvas.winfo_width(), self.canvas.winfo_height())
        else:
            self.view.constrain(self.canvas.winfo_width(), self.canvas.winfo_height())
        self._queue_draw()

    def _wheel(self, event):
        if self.image is not None:
            delta = getattr(event, "delta", 0)
            number = getattr(event, "num", None)
            steps = (1 if number == 4 else -1 if number == 5 else
                     max(-4, min(4, delta / 120 if abs(delta) >= 120 else delta)))
            if steps:
                self._fit_mode = False
                self.view.zoom_at(1.2 ** steps, event.x, event.y,
                                  self.canvas.winfo_width(), self.canvas.winfo_height())
                self._queue_draw()
        return "break"  # Do not scroll the main settings pane.

    def _start_pan(self, event):
        self._drag = (event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def _pan(self, event):
        if self.image is not None and self._drag is not None:
            self._fit_mode = False
            self.view.x += event.x - self._drag[0]
            self.view.y += event.y - self._drag[1]
            self.view.constrain(self.canvas.winfo_width(), self.canvas.winfo_height())
            self._queue_draw()
        self._drag = (event.x, event.y)

    def _end_pan(self, _event):
        self._drag = None
        self.canvas.configure(cursor="")

    def _queue_draw(self):
        if self._draw_id is None:
            self._draw_id = self.after_idle(self._render_view)

    def _render_view(self):
        self._draw_id = None
        if self.image is None:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        scale = self.view.scale
        # Only render the visible viewport, even at 1600% on a large AI result.
        try:
            visible = self.image.transform(
                (width, height), Image.Transform.AFFINE,
                (1 / scale, 0, -self.view.x / scale, 0, 1 / scale, -self.view.y / scale),
                resample=Image.Resampling.BICUBIC,
            ).convert("RGBA")
            self._photo = ImageTk.PhotoImage(visible, master=self.canvas)
            self.canvas.itemconfigure(self._image_item, image=self._photo)
            self._on_zoom(f"{scale:.0%} · {self.view.width} × {self.view.height}")
        except (ValueError, OSError, MemoryError) as error:
            self._on_error(str(error))

    def destroy(self):
        if self._draw_id is not None:
            self.after_cancel(self._draw_id)
            self._draw_id = None
        self.image = self._photo = None
        super().destroy()


def _timestamp(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02}:{seconds:06.3f}"


class UpscalePreview(CTkToplevel):
    """Tk owns controls/results; a single spawn worker owns decoding and AI."""

    def __init__(self, master, paths, get_config, on_close):
        super().__init__(master)
        self.title("Preview upscale")
        self.geometry("1000x760")
        self.minsize(640, 420)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._get_config = get_config
        self._on_close = on_close
        self._paths = list(paths)
        self._labels = [f"{index + 1}: {Path(path).name}" for index, path in enumerate(paths)]
        self._source = None
        self._index = 0
        self._generation = 0
        self._pending = None
        self._inflight = False
        self._seek_id = self._poll_id = None
        self._closed = False
        self._failed = False
        self._settings_key = None
        self._worker = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        header = CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        header.grid_columnconfigure(0, weight=1)
        self.files = CTkOptionMenu(header, values=self._labels, command=self._select_file,
                                   dynamic_resizing=False)
        self.files.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        CTkButton(header, text="Refresh", width=80, command=self.refresh).grid(
            row=0, column=1, padx=8, pady=8)
        self.viewport = ImageViewport(
            self, on_zoom=lambda text: self.zoom_label.configure(text=text),
            on_error=lambda text: self.status.configure(text=f"Preview display error: {text}"),
        )
        self.viewport.grid(row=1, column=0, sticky="nsew", padx=12)
        controls = CTkFrame(self)
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=12)
        controls.grid_columnconfigure(1, weight=1)
        self.previous = CTkButton(controls, text="◀", width=36, command=lambda: self._step(-1))
        self.previous.grid(row=0, column=0, padx=8, pady=8)
        self.seek = CTkSlider(controls, from_=0, to=1, number_of_steps=1,
                              command=self._seek, state="disabled")
        self.seek.set(0)
        self.seek.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        self.next = CTkButton(controls, text="▶", width=36, command=lambda: self._step(1))
        self.next.grid(row=0, column=2, padx=8, pady=8)
        self.position = CTkLabel(controls, text="Reading source…")
        self.position.grid(row=1, column=0, columnspan=3, pady=(0, 4))
        CTkButton(controls, text="Fit", width=60, command=self.viewport.fit).grid(
            row=2, column=0, padx=8, pady=4)
        self.zoom_label = CTkLabel(controls, text="Scroll to zoom · Drag to pan")
        self.zoom_label.grid(row=2, column=1, padx=8, pady=4)
        CTkButton(controls, text="100%", width=60, command=self.viewport.actual_size).grid(
            row=2, column=2, padx=8, pady=4)
        self.status = CTkLabel(
            self, text="", wraplength=600, anchor="w", justify="left",
        )
        self.status.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.bind("<Configure>", self._wrap_status, add="+")
        try:
            self._worker = PreviewWorker()
            self._select_file(self._labels[0])
            self._poll_id = self.after(50, self._poll)
        except Exception:
            self.destroy()
            raise

    def _wrap_status(self, event):
        if event.widget is self:
            self.status.configure(wraplength=max(200, self.winfo_width() - 40))

    def _select_file(self, label):
        if self._closed or self._failed:
            return
        self._path = self._paths[self._labels.index(label)]
        self._source = None
        self._index = 0
        self._settings_key = None
        self.viewport.clear()
        self.seek.set(0)
        self._set_seek_state(False)
        self.position.configure(text="Reading source…")
        self._queue(PreviewRequest(self._generation + 1, self._path))

    def _set_seek_state(self, enabled):
        state = "normal" if enabled else "disabled"
        for widget in (self.seek, self.previous, self.next):
            widget.configure(state=state)

    def _show_position(self):
        source = self._source
        if source.is_video:
            text = (f"Frame {self._index + 1} / {source.frame_count} · "
                    f"{_timestamp(self._index / source.fps)} / "
                    f"{_timestamp((source.frame_count - 1) / source.fps)}")
        else:
            text = "Still image"
        self.position.configure(text=text)

    def _seek(self, value):
        if self._source is None:
            return
        self._index = max(0, min(round(float(value)), self._source.frame_count - 1))
        self.seek.set(self._index)
        self._show_position()
        self.refresh()

    def _step(self, delta):
        self._seek(self._index + delta)

    def refresh(self):
        if self._closed or self._failed:
            return
        if self._source is None:
            self._queue(PreviewRequest(self._generation + 1, self._path))
            return
        try:
            config = self._get_config()
        except ValueError as error:
            self._generation += 1
            self._settings_key = None
            self._pending = None
            self.viewport.clear()
            self.status.configure(text=str(error))
            return
        self._settings_key = self._config_key(config)
        self._queue(PreviewRequest(self._generation + 1, self._path, self._index, config))

    @staticmethod
    def _config_key(config):
        return (config.selected_AI_model, config.selected_gpu, config.input_resize_factor,
                config.output_resize_factor, config.tiles_resolution,
                config.selected_sharpening_amount)

    def settings_changed(self):
        try:
            key = self._config_key(self._get_config())
        except ValueError:
            key = None
        if key != self._settings_key:
            self.refresh()

    def _queue(self, request):
        if self._closed or self._failed:
            return
        self._generation = request.generation
        self._pending = request
        self.status.configure(text="Upscaling preview…" if request.config else "Reading source…")
        # Never imply the previous frame/settings are the requested result.
        self.viewport.clear()
        if self._seek_id is not None:
            self.after_cancel(self._seek_id)
        self._seek_id = self.after(180, self._dispatch)

    def _dispatch(self):
        self._seek_id = None
        if not (self._closed or self._failed) and self._pending is not None and not self._inflight:
            self._worker.submit(self._pending)
            self._pending = None
            self._inflight = True

    def _poll(self):
        self._poll_id = None
        try:
            result = self._worker.receive()
        except RuntimeError as error:
            self._failed = True
            self._inflight = False
            self._pending = None
            if self._seek_id is not None:
                self.after_cancel(self._seek_id)
                self._seek_id = None
            self.status.configure(text=f"Preview error: {error}. Close and reopen to retry.")
            return
        if result is not None:
            self._inflight = False
            if result.generation == self._generation:
                if result.error:
                    self.status.configure(text=f"Preview error: {result.error}")
                else:
                    self._source = result.source
                    self._index = result.frame_index
                    last = max(1, self._source.frame_count - 1)
                    self.seek.configure(to=last, number_of_steps=last)
                    self.seek.set(self._index)
                    self._set_seek_state(self._source.is_video and self._source.frame_count > 1)
                    self._show_position()
                    if result.image is None:
                        self.refresh()
                    else:
                        try:
                            self.viewport.set_image(result.image)
                        except Exception as error:
                            self.status.configure(text=f"Preview display error: {error}")
                        else:
                            self.status.configure(
                                text="Upscaled · Pre-encode preview · Scroll to zoom · Drag to pan",
                            )
            if self._seek_id is None:
                self._dispatch()
        self._poll_id = self.after(50, self._poll)

    def destroy(self):
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        for handle in (self._seek_id, self._poll_id):
            if handle is not None:
                try:
                    self.after_cancel(handle)
                except TclError:
                    pass
        self._seek_id = self._poll_id = None
        self._on_close()
        if hasattr(self, "viewport"):
            self.viewport.destroy()
        super().destroy()
