"""Spawn-safe preview service. Only this process owns the preview AI session."""

from dataclasses import dataclass
from multiprocessing import get_context
from queue import Empty, Queue
from threading import Event, Thread

import cv2
import numpy as np
from PIL import Image

from .config import ProcessingConfig


@dataclass
class PreviewRequest:
    generation: int
    path: str
    frame_index: int = 0
    config: ProcessingConfig | None = None  # None probes metadata without inference.


@dataclass
class PreviewResult:
    generation: int
    source: object = None
    frame_index: int = 0
    image: Image.Image | None = None
    error: str | None = None


def _display_image(image):
    """Prepare premultiplied alpha off Tk, including high-bit-depth input."""
    if image.dtype != np.uint8:
        image = np.clip(image.astype(np.float32) / 257, 0, 255).astype(np.uint8)
    conversion = (cv2.COLOR_GRAY2RGBA if image.ndim == 2 else
                  cv2.COLOR_BGRA2RGBA if image.shape[2] == 4 else cv2.COLOR_BGR2RGBA)
    # Pillow's RGBA bicubic transform first premultiplies the entire source.
    # Do that once here, not on every Tk zoom/pan/resize of a large upscale.
    return Image.fromarray(cv2.cvtColor(image, conversion)).convert("RGBa")


def run_preview_worker(requests, results):
    # Never import Tk in a spawned inference process.
    from .preview import create_preview_engine, probe_source, render_preview

    source = engine = engine_key = None
    while True:
        request = requests.get()
        if request is None:
            return
        image = None
        try:
            if source is None or source.path != request.path:
                source = probe_source(request.path)
            index = max(0, min(request.frame_index, source.frame_count - 1))
            if request.config is not None:
                config = request.config
                key = (config.selected_AI_model, config.selected_gpu,
                       config.input_resize_factor, config.tiles_resolution)
                if key != engine_key:
                    # Release the old GPU session before allocating its replacement.
                    engine = None
                    engine_key = None
                    engine = create_preview_engine(config)
                    engine_key = key
                image = _display_image(render_preview(source, index, config, engine=engine))
            result = PreviewResult(request.generation, source, index, image)
        except Exception as error:
            result = PreviewResult(request.generation, error=str(error))
        results.send(result)
        # Do not retain a full previous display buffer during the next inference.
        result = image = None


class PreviewWorker:
    """One in-flight request; the UI retains/coalesces the newest pending seek."""

    def __init__(self):
        context = get_context("spawn")
        self.requests = context.Queue(maxsize=1)
        self._receiver, sender = context.Pipe(duplex=False)
        self._completed = Queue()
        self._closed = Event()
        self._reader = None
        self.process = context.Process(
            target=run_preview_worker, args=(self.requests, sender),
            name="QualityScaler-preview", daemon=True,
        )
        try:
            self.process.start()
            # Only the child retains a writer, so even a native crash/kill
            # unblocks the reader with EOF instead of leaving a partial receive.
            sender.close()
            self._reader = Thread(
                target=self._receive_results, name="QualityScaler-preview-results", daemon=True,
            )
            self._reader.start()
        except Exception:
            sender.close()
            self.close()
            raise

    def _receive_results(self):
        # A multiprocessing get_nowait/recv can block on a partially transferred
        # large image. Unpickle pixels here, never in Tk's event loop. PIL's raw
        # image transport also avoids PNG decoding/decompression-bomb limits.
        try:
            while not self._closed.is_set():
                result = self._receiver.recv()
                if not self._closed.is_set():
                    self._completed.put(result)
                result = None
        except Exception as error:
            if not self._closed.is_set():
                self._completed.put(RuntimeError(f"Preview worker connection closed: {error}"))

    def submit(self, request):
        self.requests.put_nowait(request)

    def receive(self):
        try:
            result = self._completed.get_nowait()
        except Empty:
            return None
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        # Preview writes no files/checkpoints. Cancel even a stuck decoder or
        # native inference call instead of holding up Tk/application shutdown.
        if self.process.pid is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=0.2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=0.2)
        if self._reader is not None:
            self._reader.join(timeout=0.2)
        self._receiver.close()
        self.requests.cancel_join_thread()
        self.requests.close()
