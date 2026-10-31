"""Resumable video jobs with one SQLite cache and bounded in-flight frames."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from multiprocessing import get_context
from pathlib import Path
from threading import Event, local
from time import monotonic

import cv2
import numpy as np

from .ai import AI_upscale, calculate_optimal_workers
from .ffmpeg import FFmpeg, UpscaleCancelled
from .frame_store import FrameStore
from .job_lock import JobLock
from .media import blend_images, copy_file_metadata, get_video_fps
from .paths import prepare_output_video_directory_name, prepare_output_video_filename


@dataclass(frozen=True)
class VideoSettings:
    model: str
    gpu: str = "Auto"
    input_resize_factor: float = 1.0
    output_resize_factor: float = 1.0
    tiles_resolution: int = 1024
    workers: int = 1
    blending_factor: float = 0.0
    codec: str = "x264"
    extension: str = ".mp4"
    keep_frames: bool = False

    def __post_init__(self):
        factors = (self.input_resize_factor, self.output_resize_factor)
        if any(not math.isfinite(value) or value <= 0 for value in factors):
            raise ValueError("Resize factors must be finite and positive")
        if self.workers < 1 or self.tiles_resolution < 1:
            raise ValueError("Worker count and tile resolution must be positive")
        if not 0 <= self.blending_factor <= 1:
            raise ValueError("Blending factor must be between 0 and 1")
        if self.extension not in (".mp4", ".mkv", ".avi", ".mov"):
            raise ValueError(f"Unsupported video extension: {self.extension}")

    def processing_settings(self):
        # Execution and encoding choices can change without invalidating JPEGs.
        return {key: asdict(self)[key] for key in (
            "model", "input_resize_factor", "output_resize_factor",
            "tiles_resolution", "blending_factor",
        )}


def create_engine(settings):
    return AI_upscale(
        settings.model, settings.gpu, settings.input_resize_factor,
        settings.output_resize_factor, settings.tiles_resolution,
    )


def decode_frame(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Cached frame is not a decodable JPEG")
    return image


_worker_state = local()


def initialize_worker(settings, engine_factory):
    """One worker owns one model session for its entire lifetime."""
    _worker_state.engine = engine_factory(settings)
    _worker_state.blending_factor = settings.blending_factor


def process_frame(data):
    """Decode, infer, blend and encode without opening SQLite in a worker."""
    original = decode_frame(data)
    upscaled = _worker_state.engine.AI_orchestration(original)
    if _worker_state.blending_factor:
        upscaled = blend_images(original, upscaled, _worker_state.blending_factor)
    success, encoded = cv2.imencode(".jpg", upscaled)
    if not success:
        raise ValueError("Could not encode upscaled frame")
    return encoded.tobytes()


def create_executor(**kwargs):
    # ORT's Python DirectML allocator is process-global and unsafe across
    # concurrent sessions in threads. Spawn also avoids inheriting GPU state.
    # https://github.com/microsoft/onnxruntime/issues/20713
    return ProcessPoolExecutor(mp_context=get_context("spawn"), **kwargs)


class VideoUpscaler:
    """Own the job lifecycle, SQLite writer and bounded inference workers.

    Spawned workers have independent AI sessions and return encoded JPEGs. Only
    this object's calling thread accesses SQLite; no database connection or
    decoded image is sent across processes, and progress counts *saved* frames.
    """

    def __init__(self, *, ffmpeg=None, engine_factory=create_engine, status=None,
                 stop_event=None, checkpoint_interval=16, executor_factory=create_executor):
        if checkpoint_interval < 1:
            raise ValueError("Checkpoint interval must be positive")
        self.ffmpeg = ffmpeg if ffmpeg is not None else FFmpeg()
        self.engine_factory = engine_factory
        self.executor_factory = executor_factory
        self.status = status if status is not None else lambda message: None
        self.stop_event = stop_event if stop_event is not None else Event()
        self.checkpoint_interval = checkpoint_interval

    def _check_stop(self):
        if self.stop_event.is_set():
            raise UpscaleCancelled()

    @staticmethod
    def _state(store, stage, **metadata):
        store.set_metadata("stage", stage)
        store.set_metadata("updated_at", datetime.now(timezone.utc).isoformat())
        for key, value in metadata.items():
            store.set_metadata(key, value)
        store.commit()

    @staticmethod
    def _prepare_job(store, video_path, output_path, settings):
        source_path = Path(video_path).resolve()
        stat = source_path.stat()
        source = {
            "path": str(source_path), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        previous_source = store.get_metadata("source")
        if previous_source is not None:
            if store.get_metadata("job_version") != 1:
                raise ValueError("Unsupported cached job version")
            if previous_source != source:
                raise ValueError(
                    "The cached job belongs to a different or modified source. "
                    "Choose another output folder or move the old .qualityscaler cache."
                )
            if store.get_metadata("processing_settings") != settings.processing_settings():
                raise ValueError(
                    "The cached job has different AI, resize, tile, or blending settings. "
                    "Restore its recorded settings or choose another output folder. "
                    "The existing cache has not been changed."
                )
        elif store.frame_count("source") or store.frame_count("upscaled"):
            raise ValueError("Frame cache is missing its job identity; refusing to mix frames")
        store.set_metadata("job_version", 1)
        store.set_metadata("source", source)
        store.set_metadata("processing_settings", settings.processing_settings())
        store.set_metadata("settings", asdict(settings))
        store.set_metadata("output_path", str(Path(output_path).resolve()))
        store.commit()

    def _extract(self, store, video_path):
        if store.get_metadata("extraction_complete", False):
            return
        self._state(store, "extracting")
        self.status("Extracting video frames")
        fps = get_video_fps(str(video_path))
        previous_count = store.frame_count("source")

        def save_frame(index, data):
            self._check_stop()
            # A partial extraction is replayed, but committed frames are reused.
            # Seeking by time could skip/duplicate frames at keyframe boundaries.
            if not store.has_frame("source", index):
                store.put_frame("source", index, data)
            if (index + 1) % self.checkpoint_interval == 0:
                store.commit()
                self.status(f"Extracting video frames ({index + 1} saved)")

        count = self.ffmpeg.extract(video_path, fps, self.stop_event, save_frame)
        self._check_stop()
        if count < previous_count or count != store.frame_count("source"):
            raise ValueError("Extracted frame count does not match the cached source")
        self._state(store, "upscaling", fps=fps, total_frames=count,
                    extraction_complete=True)

    def _upscale(self, store, settings):
        total = store.get_metadata("total_frames")
        if not isinstance(total, int) or total <= 0 or store.frame_count("source") != total:
            raise ValueError("Source frame cache is incomplete")
        completed = store.frame_count("upscaled")
        if completed == total:
            return
        if completed > total:
            raise ValueError("Upscaled frame cache contains unexpected frames")
        self._state(store, "upscaling")
        self._check_stop()
        workers = max(1, min(
            settings.workers, total - completed,
            calculate_optimal_workers(
                decode_frame(store.get_frame("source", 0)), settings.model,
                settings.input_resize_factor, settings.tiles_resolution, settings.workers,
            ),
        ))

        # Submit at most 2 * workers, not one Future (or ndarray) per video frame.
        # Source and result data are bounded even for million-frame videos.
        missing = (index for index in range(total)
                   if not store.has_frame("upscaled", index))
        pending = {}
        started = monotonic()
        saved_this_run = 0
        with self.executor_factory(
            max_workers=workers, initializer=initialize_worker,
            initargs=(settings, self.engine_factory),
        ) as executor:
            try:
                def fill_pending():
                    while len(pending) < workers * 2:
                        self._check_stop()
                        index = next(missing, None)
                        if index is None:
                            break
                        future = executor.submit(
                            process_frame, store.get_frame("source", index)
                        )
                        pending[future] = index

                fill_pending()
                while pending:
                    self._check_stop()
                    done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        store.put_frame("upscaled", index, future.result())
                        completed += 1
                        saved_this_run += 1
                        if saved_this_run % self.checkpoint_interval == 0:
                            store.commit()
                        seconds_per_frame = (monotonic() - started) / saved_this_run
                        remaining = round(seconds_per_frame * (total - completed))
                        self.status(
                            f"Upscaling video {completed / total:.2%} "
                            f"({completed}/{total} saved, {remaining}s remaining)"
                        )
                    fill_pending()
            finally:
                for future in pending:
                    future.cancel()
                # Publish already-saved results before waiting for a running
                # inference to return. A later hard kill need not lose them.
                store.commit()
        store.commit()
        if store.frame_count("upscaled") != total:
            raise ValueError("Not all video frames were saved")

    def run(self, video_path, output_path, cache_path, settings):
        # SQLite transactions protect each batch, not the lifetime between
        # batches. Hold ownership through metadata preparation and cache removal.
        with JobLock(cache_path):
            return self._run_locked(video_path, output_path, cache_path, settings)

    def _run_locked(self, video_path, output_path, cache_path, settings):
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with FrameStore(cache_path) as store:
            # Identity mismatches must not change the existing job's state.
            self._prepare_job(store, video_path, output_path, settings)
            try:
                self._check_stop()
                self._extract(store, video_path)
                self._upscale(store, settings)
                self._check_stop()
                self._state(store, "encoding", error=None)
                self.status("Encoding upscaled video")
                self.ffmpeg.encode(
                    video_path, output_path, store.get_metadata("fps"),
                    settings.codec, store.iter_frames("upscaled"), self.stop_event,
                )
                self._check_stop()
                copy_file_metadata(str(video_path), str(output_path))
                self._state(store, "complete")
            except BaseException as error:
                try:
                    self._state(
                        store, "paused" if isinstance(error, UpscaleCancelled) else "failed",
                        interrupted_stage=store.get_metadata("stage"),
                        error=str(error) or type(error).__name__,
                    )
                except Exception:
                    # A disk-full/SQLite error must not hide the original failure.
                    # SQLite rolls back an uncommitted batch on the next open.
                    pass
                raise
        if not settings.keep_frames:
            cache_path.unlink()
            try:
                cache_path.parent.rmdir()
            except OSError:
                pass  # Never delete unrelated files in a cache directory.
        return str(output_path)


def upscale_video(
    process_status_q, event_stop_upscale_process, video_path, file_number,
    selected_output_path, selected_AI_model, selected_blending_factor,
    selected_AI_multithreading, selected_gpu, input_resize_factor,
    output_resize_factor, tiles_resolution, selected_video_extension,
    selected_video_codec, selected_keep_frames,
):
    """GUI adapter; backend modules never import the GUI entry point."""
    from .status import write_process_status

    settings = VideoSettings(
        model=selected_AI_model, gpu=selected_gpu,
        input_resize_factor=input_resize_factor, output_resize_factor=output_resize_factor,
        tiles_resolution=tiles_resolution, workers=selected_AI_multithreading,
        blending_factor=selected_blending_factor, codec=selected_video_codec,
        extension=selected_video_extension, keep_frames=selected_keep_frames,
    )
    directory = prepare_output_video_directory_name(
        video_path, selected_output_path, selected_AI_model,
        input_resize_factor, output_resize_factor, selected_blending_factor,
    )
    output = prepare_output_video_filename(
        video_path, selected_output_path, selected_AI_model,
        input_resize_factor, output_resize_factor, selected_video_extension,
        selected_blending_factor,
    )
    return VideoUpscaler(
        stop_event=event_stop_upscale_process,
        status=lambda message: write_process_status(
            process_status_q, f"{file_number}. {message}"
        ),
    ).run(video_path, output, Path(directory) / "frames.sqlite3", settings)
