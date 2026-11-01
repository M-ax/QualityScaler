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
    def _pipe(self, arguments, stop_event, *, writing=False, progress=None):
        if stop_event.is_set():
            raise UpscaleCancelled()
        process = subprocess.Popen(
            [self.executable, "-hide_banner", "-loglevel", "error", *arguments],
            stdin=subprocess.PIPE if writing else subprocess.DEVNULL,
            stdout=subprocess.PIPE if not writing or progress else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS)
            if sys.platform == "win32" else 0,
        )
        # Drain stderr continuously to avoid a full-pipe deadlock, retaining at
        # most 64 KiB even if a damaged video emits diagnostics for every frame.
        errors = deque(maxlen=8)
        progress_errors = []
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
        def drain_progress():
            for line in process.stdout:
                if line.startswith(b"frame="):
                    try:
                        frame = int(line.partition(b"=")[2])
                    except ValueError:
                        continue
                    try:
                        progress(frame)
                    except Exception as error:
                        # Keep draining even if a UI/status consumer fails,
                        # otherwise a full progress pipe could deadlock FFmpeg.
                        if not progress_errors:
                            progress_errors.append(error)

        progress_reader = Thread(target=drain_progress, daemon=True) if writing and progress else None
        watcher = Thread(target=watch_cancellation, daemon=True)
        error_reader.start()
        if progress_reader:
            progress_reader.start()
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
            if progress_reader:
                progress_reader.join()
                if progress_errors:
                    raise progress_errors[0]
        finally:
            self._terminate(process)
            try:
                pipe.close()
            except BrokenPipeError:
                pass  # Preserve the FFmpeg diagnostic raised above.
            finished.set()
            watcher.join()
            error_reader.join()
            if progress_reader:
                progress_reader.join()
                process.stdout.close()
            process.stderr.close()

    def extract(self, video_path, fps, stop_event, save_frame, *, input_size=None):
        """Pass each complete JPEG to save_frame(index, bytes) in display order.

        An explicit (width, height) tuple scales frames before JPEG encoding.
        Dimensions must be even integers >= 2. None preserves full-resolution
        extraction, including the source dimensions of resumable legacy jobs.
        """
        filters = f"fps={fps}"
        if input_size is not None:
            if not isinstance(input_size, tuple):
                raise TypeError("input_size must be a (width, height) tuple")
            if len(input_size) != 2:
                raise ValueError("input_size must contain exactly (width, height)")
            for dimension in input_size:
                if isinstance(dimension, bool) or not isinstance(dimension, int):
                    raise TypeError("input_size dimensions must be integers (not bools)")
                if dimension < 2 or dimension % 2:
                    raise ValueError("input_size dimensions must be even integers >= 2")
            filters += f",scale={input_size[0]}:{input_size[1]}:flags=area"
        arguments = [
            "-threads", "0", "-err_detect", "ignore_err", "-hwaccel", "auto",
            "-i", str(video_path), "-map", "0:v:0", "-an",
            "-vf", filters, "-qscale:v", "3", "-c:v", "mjpeg",
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

    def encode(self, video_path, output_path, fps, codec, frames, stop_event, *,
               output_size=None, total_frames=None, status=None):
        """Encode ordered cached JPEGs, preferring passthrough for optional audio.

        Replayable frames allow video-codec and AAC audio fallbacks. Publish only
        a successful encode; never replace an existing output with a failed video.
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
        codecs = [codec] if codec == "libx264" else [codec, "libx264"]
        filters = "format=yuv420p"
        if output_size is not None:
            filters = f"scale={output_size[0]}:{output_size[1]}," + filters
        # A callable supplies a fresh ordered SQLite iterator for codec retries.
        # Sequences are naturally replayable; a bare iterator is single-use.
        if callable(frames):
            frame_factory = frames
            replayable = True
        elif iter(frames) is frames:
            frame_factory = lambda: frames
            codecs = codecs[:1]
            replayable = False
        else:
            frame_factory = lambda: iter(frames)
            replayable = True
        attempts = [(video_codec, "copy") for video_codec in codecs]
        if replayable:
            # WMV commonly contains WMA audio, which cannot be copied into MP4
            # or MOV. Preserve passthrough when possible; retry with AAC rather
            # than dropping audio or failing a valid video after upscaling.
            attempts += [(video_codec, "aac") for video_codec in codecs]

        def progress(count):
            percent = min(100, int(count * 100 / total_frames)) if total_frames else 0
            status(f"Encoding upscaled video {percent}%")

        try:
            for index, (current_codec, audio_codec) in enumerate(attempts):
                arguments = [
                    "-y", "-f", "image2pipe", "-framerate", str(fps),
                    "-c:v", "mjpeg", "-i", "pipe:0", "-i", str(video_path),
                    "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1",
                    "-c:v", current_codec, "-c:a", audio_codec, "-g", str(max(1, round(fps))),
                    "-vf", filters, "-color_range", "tv", "-b:v", "50000k",
                ]
                if audio_codec == "aac":
                    arguments += ["-b:a", "192k"]
                if output.suffix.lower() in (".mp4", ".mov", ".m4v"):
                    arguments += ["-movflags", "+faststart"]
                if status:
                    arguments += ["-progress", "pipe:1", "-nostats"]
                arguments.append(str(partial))
                if index and current_codec != codec and status:
                    status("Encoding upscaled video (x264 fallback)")
                if audio_codec == "aac" and status:
                    status("Encoding upscaled video (AAC audio fallback)")
                try:
                    with self._pipe(arguments, stop_event, writing=True,
                                    progress=progress if status else None) as pipe:
                        for frame in frame_factory():
                            if stop_event.is_set():
                                raise UpscaleCancelled()
                            pipe.write(frame)
                    if stop_event.is_set():
                        raise UpscaleCancelled()
                    os.replace(partial, output)
                    return
                except RuntimeError:
                    if index == len(attempts) - 1:
                        raise
        finally:
            partial.unlink(missing_ok=True)
