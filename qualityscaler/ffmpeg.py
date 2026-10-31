"""Streaming FFmpeg I/O: no per-frame files or concat manifests."""

from collections import deque
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
from threading import Event, Thread

from .jpeg_stream import iter_jpeg_frames
from .paths import find_by_relative_path


class UpscaleCancelled(Exception):
    """The user stopped a job; its committed cache can be resumed."""


class FFmpeg:
    def __init__(self, executable=None):
        bundled = find_by_relative_path("Assets/ffmpeg.exe")
        self.executable = str(executable or (
            bundled if Path(bundled).is_file() else shutil.which("ffmpeg") or bundled
        ))

    @staticmethod
    def _terminate(process):
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                if process.poll() is None:
                    raise
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    @contextmanager
    def _pipe(self, arguments, stop_event, *, writing=False):
        if stop_event.is_set():
            raise UpscaleCancelled()
        process = subprocess.Popen(
            [self.executable, "-hide_banner", "-loglevel", "error", *arguments],
            stdin=subprocess.PIPE if writing else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL if writing else subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        # Drain stderr continuously to avoid a full-pipe deadlock, retaining at
        # most 64 KiB even if a damaged video emits diagnostics for every frame.
        errors = deque(maxlen=8)
        truncated = Event()
        finished = Event()

        def drain_errors():
            while True:
                data = process.stderr.read(8192)
                if not data:
                    return
                if len(errors) == errors.maxlen:
                    truncated.set()
                errors.append(data)

        def diagnostic():
            error_reader.join()
            detail = b"".join(errors).decode("utf-8", errors="replace").strip()
            prefix = "[Earlier FFmpeg diagnostics omitted]\n" if truncated.is_set() else ""
            return prefix + detail

        def watch_cancellation():
            while not finished.wait(0.1):
                if stop_event.is_set():
                    self._terminate(process)
                    return

        error_reader = Thread(target=drain_errors, daemon=True)
        watcher = Thread(target=watch_cancellation, daemon=True)
        error_reader.start()
        watcher.start()
        pipe = process.stdin if writing else process.stdout
        try:
            try:
                yield pipe
                pipe.close()
                returncode = process.wait()
            except BaseException as error:
                self._terminate(process)
                if stop_event.is_set():
                    raise UpscaleCancelled() from error
                if isinstance(error, BrokenPipeError):
                    raise RuntimeError(f"FFmpeg failed: {diagnostic()}") from error
                raise
            if stop_event.is_set():
                raise UpscaleCancelled()
            if returncode:
                raise RuntimeError(f"FFmpeg failed ({returncode}): {diagnostic()}")
        finally:
            self._terminate(process)
            try:
                pipe.close()
            except BrokenPipeError:
                pass  # Preserve the FFmpeg diagnostic raised above.
            finished.set()
            watcher.join()
            error_reader.join()
            process.stderr.close()

    def extract(self, video_path, fps, stop_event, save_frame):
        """Pass each complete JPEG to save_frame(index, bytes) in display order."""
        arguments = [
            "-i", str(video_path), "-map", "0:v:0", "-an",
            "-r", str(fps), "-qscale:v", "3", "-c:v", "mjpeg",
            "-f", "image2pipe", "pipe:1",
        ]
        count = 0
        with self._pipe(arguments, stop_event) as pipe:
            for index, frame in enumerate(iter_jpeg_frames(pipe)):
                if stop_event.is_set():
                    raise UpscaleCancelled()
                save_frame(index, frame)
                count += 1
        if not count:
            raise ValueError("FFmpeg extracted no video frames")
        return count

    def encode(self, video_path, output_path, fps, codec, frames, stop_event):
        """Encode ordered cached JPEGs and copy optional audio in one invocation.

        Publish only a successful encode; an existing output is never replaced
        with a partial or failed video.
        """
        if stop_event.is_set():
            raise UpscaleCancelled()
        output = Path(output_path)
        # Reserve an owned sibling file, never a predictable path that might
        # already be somebody's source video or another runner's temporary file.
        descriptor, partial_name = tempfile.mkstemp(
            prefix=f".{output.stem}.partial-", suffix=output.suffix, dir=output.parent
        )
        os.close(descriptor)
        partial = Path(partial_name)
        codec = {"x264": "libx264", "x265": "libx265"}.get(codec, codec)
        arguments = [
            "-y", "-f", "image2pipe", "-framerate", str(fps),
            "-c:v", "mjpeg", "-i", "pipe:0", "-i", str(video_path),
            "-map", "0:v:0", "-map", "1:a?",
            "-map_metadata", "1", "-c:v", codec,
            "-vf", "scale=in_range=full:out_range=limited,format=yuv420p",
            "-color_range", "tv", "-b:v", "12000k", "-c:a", "copy",
            str(partial),
        ]
        try:
            with self._pipe(arguments, stop_event, writing=True) as pipe:
                for frame in frames:
                    if stop_event.is_set():
                        raise UpscaleCancelled()
                    pipe.write(frame)
            if stop_event.is_set():
                raise UpscaleCancelled()
            os.replace(partial, output)
        finally:
            partial.unlink(missing_ok=True)
