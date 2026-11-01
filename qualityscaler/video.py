"""Resumable video jobs with one SQLite cache and bounded in-flight frames."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import math
from multiprocessing import current_process, get_context
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Event, local

import cv2
import numpy as np

from .ai import AI_upscale, calculate_optimal_workers, get_model_upscale_factor
from .ffmpeg import FFmpeg, UpscaleCancelled
from .frame_store import FrameStore
from .geometry import VideoGeometry
from .job_lock import JobLock
from .media import blend_images, get_video_fps, sharpen_image
from .paths import find_by_relative_path, prepare_output_video_directory_name, prepare_output_video_filename
from .status import PipelineStatus, VideoProgress


SOURCE_LAYOUT_ORIGINAL = "original"
SOURCE_LAYOUT_SCALED = "ffmpeg-area-v1"
CURRENT_JOB_VERSION = 3


class CacheSettingsMismatch(ValueError):
    """A known source/layout cache needs explicit permission to change settings."""


def source_frame_layout(store):
    # Untagged version-2 jobs contain full-resolution source JPEGs. Keep their
    # original processing path so partial extraction/inference can still resume.
    layout = store.get_metadata("source_frame_layout", SOURCE_LAYOUT_ORIGINAL)
    version = store.get_metadata("job_version", 2)
    if (version, layout) not in ((2, SOURCE_LAYOUT_ORIGINAL), (3, SOURCE_LAYOUT_SCALED)):
        raise ValueError(f"Unsupported cached source frame layout: {layout!r}")
    return layout


def source_frame_size(store):
    size = store.get_metadata("source_size")
    if (not isinstance(size, list) or len(size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size)):
        raise ValueError("Scaled frame cache is missing valid original video dimensions")
    return tuple(size)


def geometry_for_settings(size, settings):
    return VideoGeometry.calculate(
        *size, settings.input_resize_factor, get_model_upscale_factor(settings.model),
        settings.output_resize_factor,
    )


@lru_cache(maxsize=32)
def _model_digest(path, size, mtime_ns):
    # Cache by the local file revision, but persist only its content hash: a
    # onefile executable extracts identical models to a new directory each run.
    with open(path, "rb") as model:
        return hashlib.file_digest(model, "sha256").hexdigest()


def model_digest(name):
    path = Path(find_by_relative_path(f"AI-onnx/{name}_fp16.onnx"))
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return _model_digest(str(path), stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class VideoSettings:
    model: str
    gpu: str = "Auto"
    input_resize_factor: float = 1.0
    output_resize_factor: float = 1.0
    tiles_resolution: int = 1024
    workers: int = 1
    blending_factor: float = 0.0
    sharpening_amount: float = 0.0
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
        if not math.isfinite(self.sharpening_amount) or self.sharpening_amount < 0:
            raise ValueError("Sharpening amount must be finite and nonnegative")
        if self.extension not in (".mp4", ".mkv", ".avi", ".mov"):
            raise ValueError(f"Unsupported video extension: {self.extension}")

    def processing_settings(self):
        # Execution and encoding choices can change without invalidating JPEGs.
        result = {key: asdict(self)[key] for key in (
            "model", "input_resize_factor", "output_resize_factor",
            "tiles_resolution", "blending_factor", "sharpening_amount",
        )}
        result["engine"] = "2026.4"
        result["model_sha256"] = model_digest(self.model)
        return result


def create_engine(settings):
    return AI_upscale(
        settings.model, settings.gpu, settings.input_resize_factor, settings.tiles_resolution,
    )


def decode_frame(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Cached frame is not a decodable JPEG")
    return image


_worker_state = local()


def initialize_worker(settings, engine_factory):
    """One worker owns one model session for its entire lifetime."""
    if sys.platform == "win32" and current_process().name != "MainProcess":
        import psutil
        try:
            psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
        except psutil.Error:
            pass
    cv2.setNumThreads(1)
    _worker_state.engine = engine_factory(settings)
    _worker_state.blending_factor = settings.blending_factor
    _worker_state.sharpening_amount = settings.sharpening_amount


def process_frame(data):
    """Decode, infer, blend and encode without opening SQLite in a worker."""
    original = decode_frame(data)
    upscaled = _worker_state.engine.AI_orchestration(original)
    if _worker_state.blending_factor:
        upscaled = blend_images(original, upscaled, _worker_state.blending_factor)
    if _worker_state.sharpening_amount:
        upscaled = sharpen_image(upscaled, _worker_state.sharpening_amount)
    # 2026.4 saves native AI frames, then applies output scaling during encode.
    quality = 95 if _worker_state.sharpening_amount else 90
    success, encoded = cv2.imencode(".jpg", upscaled, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Could not encode upscaled frame")
    return encoded.tobytes()


def create_executor(**kwargs):
    # ORT's Python DirectML allocator is process-global and unsafe across
    # concurrent sessions in threads. Spawn also avoids inheriting GPU state.
    # https://github.com/microsoft/onnxruntime/issues/20713
    return ProcessPoolExecutor(mp_context=get_context("spawn"), **kwargs)


class _ExtractionStop:
    """Stop the producer on either user cancellation or coordinator failure."""

    def __init__(self, user_stop):
        self.user_stop = user_stop
        self.aborted = Event()

    def is_set(self):
        return self.user_stop.is_set() or self.aborted.is_set()

    def set(self):
        self.user_stop.set()


class VideoUpscaler:
    """Own the job lifecycle, SQLite writer and bounded inference workers.

    Spawned workers have independent AI sessions and return encoded JPEGs. Only
    this object's calling thread accesses SQLite. A bounded JPEG queue lets
    extraction run in a background thread while inference processes work.
    No database connection or decoded image crosses threads/processes, and
    progress counts *saved* frames.
    """

    def __init__(self, *, ffmpeg=None, engine_factory=create_engine, status=None, progress=None,
                 stop_event=None, checkpoint_interval=16, executor_factory=create_executor):
        if checkpoint_interval < 1:
            raise ValueError("Checkpoint interval must be positive")
        self.ffmpeg = ffmpeg if ffmpeg is not None else FFmpeg()
        self.engine_factory = engine_factory
        self.executor_factory = executor_factory
        self.status = status if status is not None else lambda message: None
        self.progress = progress if progress is not None else lambda snapshot: None
        self._progress = VideoProgress("", 0, 0, None, False)
        self.stop_event = stop_event if stop_event is not None else Event()
        self.checkpoint_interval = checkpoint_interval
        self.task_status = PipelineStatus(self._report)

    def _report(self, message, **changes):
        self._progress = replace(self._progress, message=message, **changes)
        self.status(message)
        self.progress(self._progress)

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

    def _prepare_job(self, store, video_path, output_path, settings, *, clear_cache=False):
        source_path = Path(video_path).resolve()
        stat = source_path.stat()
        source = {
            "path": str(source_path), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        previous_source = store.get_metadata("source")
        if previous_source is not None:
            if store.get_metadata("job_version") not in (2, CURRENT_JOB_VERSION):
                raise ValueError(
                    "This cache was created by a different processing version. "
                    "Move it aside or choose another output folder to start a 2026.4 job."
                )
            if previous_source != source:
                raise ValueError(
                    "The cached job belongs to a different or modified source. "
                    "Choose another output folder or move the old .qualityscaler cache."
                )
            source_frame_layout(store)  # Reject unknown layouts before any writes.
        elif store.frame_count("source") or store.frame_count("upscaled"):
            raise ValueError("Frame cache is missing its job identity; refusing to mix frames")

        processing_settings = settings.processing_settings()
        if previous_source is not None:
            if store.get_metadata("processing_settings") != processing_settings:
                if not clear_cache:
                    raise CacheSettingsMismatch(
                        "The cached job has different AI, resize, tile, blending, or sharpening settings. "
                        "Restore its recorded settings or choose another output folder. "
                        "The existing cache has not been changed."
                    )
                self._check_stop()
                # Reset only this verified job's frames/metadata. The surrounding
                # FrameStore transaction publishes the reset and new identity
                # together, or rolls both back if preparation fails.
                store.reset()
                previous_source = None
        if previous_source is None:
            store.set_metadata("source_frame_layout", SOURCE_LAYOUT_SCALED)
            # Older executables reject version 3, rather than treating these
            # pre-scaled JPEGs as full-resolution frames and scaling them twice.
            store.set_metadata("job_version", CURRENT_JOB_VERSION)
        store.set_metadata("source", source)
        store.set_metadata("processing_settings", processing_settings)
        store.set_metadata("settings", asdict(settings))
        store.set_metadata("output_path", str(Path(output_path).resolve()))
        store.commit()

    def _extract(self, store, video_path, settings):
        """Save producer JPEGs on the caller; yield between frames to service AI.

        Empty-queue heartbeats let inference results be saved even when the
        decoder is slow. Closing this iterator cancels and joins the producer.
        """
        if store.get_metadata("extraction_complete", False):
            return
        fps = get_video_fps(str(video_path))
        previous_count = store.frame_count("source")
        capture = cv2.VideoCapture(str(video_path))
        try:
            expected = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            expected = int(expected) if math.isfinite(expected) and expected > 0 else None
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if all(math.isfinite(value) and value > 0 for value in (width, height)):
                width, height = round(width), round(height)
            elif source_frame_layout(store) == SOURCE_LAYOUT_SCALED:
                success, first = capture.read()
                if not success or first is None:
                    raise ValueError("Unable to determine original video dimensions")
                height, width = first.shape[:2]
        finally:
            capture.release()
        self._progress = replace(self._progress, total=expected)
        input_size = None
        if source_frame_layout(store) == SOURCE_LAYOUT_SCALED:
            geometry = geometry_for_settings((width, height), settings)
            if previous_count and source_frame_size(store) != geometry.source_size:
                raise ValueError("Cached source dimensions do not match the video")
            input_size = geometry.inference_size
            self._state(
                store, "extracting", source_size=list(geometry.source_size),
                input_size=list(input_size),
            )
            message = f"Extracting video frames ({input_size[0]}×{input_size[1]} input)"
        else:
            self._state(store, "extracting")
            message = "Extracting video frames"
        # Partial extraction is replayed from zero, including already-cached
        # frames, so its timing baseline differs from resumable inference.
        self.task_status.start("extracting", message, total=expected, estimated_total=True)

        frames = Queue(maxsize=settings.workers * 2)
        stop = _ExtractionStop(self.stop_event)

        def publish_frame(index, data):
            while not stop.is_set():
                try:
                    frames.put((index, data), timeout=0.1)
                    return
                except Full:
                    pass
            raise UpscaleCancelled()

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="frame-extraction") as producer:
            result = producer.submit(
                self.ffmpeg.extract, video_path, fps, stop, publish_frame,
                input_size=input_size,
            )
            try:
                while True:
                    try:
                        index, data = frames.get(timeout=0.05)
                    except Empty:
                        if result.done():
                            # The producer may have published its last frame
                            # between the timed-out read and done() check.
                            # Only an empty read AFTER completion is final.
                            try:
                                index, data = frames.get_nowait()
                            except Empty:
                                break
                        else:
                            self._check_stop()
                            yield None
                            continue
                    self._check_stop()
                    # Replay partial extraction without keyframe seeking, but
                    # never replace a source frame already used for inference.
                    if not store.has_frame("source", index):
                        store.put_frame("source", index, data)
                    self._progress = replace(
                        self._progress, extracted=max(previous_count, index + 1),
                    )
                    if (index + 1) % self.checkpoint_interval == 0:
                        store.commit()
                    if index == 0 or (index + 1) % self.checkpoint_interval == 0:
                        progress = f"{min(100, (index + 1) / expected * 100):.0f}% " if expected else ""
                        self.task_status.update(
                            "extracting", f"Extracting video frames {progress}({index + 1} saved)",
                            completed=index + 1,
                        )
                    yield index
                count = result.result()  # Propagate decoder errors; never encode a partial stream.
                self._check_stop()
                if count < previous_count or count != store.frame_count("source"):
                    raise ValueError("Extracted frame count does not match the cached source")
                self._state(store, "upscaling", fps=fps, total_frames=count,
                            extraction_complete=True)
                self._progress = replace(self._progress, total=count, extraction_complete=True)
                self.task_status.update(
                    "extracting", f"Extracting video frames 100% ({count} saved)",
                    completed=count, total=count, estimated_total=False,
                )
                # The probe total was only an estimate. Inference may already
                # be running; update its total without resetting its own rate.
                self.task_status.update("upscaling", total=count, estimated_total=False)
            finally:
                # Also release a producer blocked on a full queue after an AI,
                # SQLite or status-consumer failure. Do not cancel the user's
                # event on failure: the original exception must remain an error.
                stop.aborted.set()
                self.task_status.finish("extracting")

    def _upscale(self, store, settings, extraction=None):
        """Pump extraction and bounded AI futures through the sole cache writer."""
        available = store.frame_count("source")
        completed = store.frame_count("upscaled")
        self._progress = VideoProgress(
            "", available, completed, store.get_metadata("total_frames"),
            extraction is None or store.get_metadata("extraction_complete", False),
        )
        extracting = extraction is not None
        executor = None
        workers = 0
        next_index = 0
        pending = {}
        saved_this_run = 0
        try:
            while True:
                self._check_stop()
                if extracting:
                    try:
                        index = next(extraction)
                    except StopIteration:
                        extracting = False
                    else:
                        if index is not None:
                            available = max(available, index + 1)
                if not extracting:
                    total = store.get_metadata("total_frames")
                    if not isinstance(total, int) or total <= 0 or available != total:
                        raise ValueError("Source frame cache is incomplete")
                    if completed > total:
                        raise ValueError("Upscaled frame cache contains unexpected frames")
                    if completed == total:
                        break
                if executor is None and available > completed:
                    frame = decode_frame(store.get_frame("source", 0))
                    if source_frame_layout(store) == SOURCE_LAYOUT_SCALED:
                        geometry = geometry_for_settings(source_frame_size(store), settings)
                        if frame.shape[1::-1] != geometry.inference_size:
                            raise ValueError("Cached frame dimensions do not match the extraction scale")
                        # Input scaling is already baked into cached JPEGs.
                        settings = replace(settings, input_resize_factor=1.0)
                    if not extracting:
                        self._state(store, "upscaling")
                    workers = max(1, min(
                        settings.workers,
                        settings.workers if extracting else total - completed,
                        calculate_optimal_workers(
                            frame, settings.model, settings.input_resize_factor,
                            settings.tiles_resolution, settings.workers,
                        ),
                    ))
                    executor = self.executor_factory(
                        max_workers=workers, initializer=initialize_worker,
                        initargs=(settings, self.engine_factory),
                    )
                    self.task_status.start(
                        "upscaling", f"Preparing video upscaling ({workers} workers)",
                        completed=completed, total=self._progress.total,
                        estimated_total=not self._progress.extraction_complete,
                    )
                if executor is None:
                    continue
                # Scan only available indices. Never retain one Future per
                # video frame, and don't mark not-yet-extracted indices missing.
                while next_index < available and len(pending) < workers * 2:
                    self._check_stop()
                    index = next_index
                    next_index += 1
                    if not store.has_frame("upscaled", index):
                        future = executor.submit(
                            process_frame, store.get_frame("source", index)
                        )
                        pending[future] = index
                done, _ = wait(
                    pending, timeout=0 if extracting else 0.1,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    index = pending.pop(future)
                    store.put_frame("upscaled", index, future.result())
                    completed += 1
                    saved_this_run += 1
                    if saved_this_run % self.checkpoint_interval == 0:
                        store.commit()
                    total = self._progress.total
                    if total:
                        message = (
                            f"Upscaling video {min(1, completed / total):.2%} "
                            f"({completed}/{total} saved)"
                        )
                    else:
                        message = f"Upscaling video ({completed} saved)"
                    self._progress = replace(self._progress, upscaled=completed)
                    self.task_status.update("upscaling", message, completed=completed)
                if not extracting and not pending and next_index >= available:
                    break
        finally:
            for future in pending:
                future.cancel()
            try:
                # Publish saved results BEFORE waiting for decoder/model cleanup.
                store.commit()
            finally:
                try:
                    if extraction is not None:
                        extraction.close()
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True)
                    self.task_status.finish("upscaling")
        store.commit()
        if store.frame_count("upscaled") != store.get_metadata("total_frames"):
            raise ValueError("Not all video frames were saved")

    def run(self, video_path, output_path, cache_path, settings, *, clear_cache=False):
        # SQLite transactions protect each batch, not the lifetime between
        # batches. Hold ownership through metadata preparation and cache removal.
        with JobLock(cache_path):
            return self._run_locked(
                video_path, output_path, cache_path, settings, clear_cache=clear_cache,
            )

    def _run_locked(self, video_path, output_path, cache_path, settings, *, clear_cache=False):
        # A resumed/reused runner must not retain interrupted stages or rates.
        self.task_status = PipelineStatus(self._report)
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with FrameStore(cache_path) as store:
            # Identity mismatches must not change the existing job's state.
            self._prepare_job(store, video_path, output_path, settings, clear_cache=clear_cache)
            try:
                self._check_stop()
                with closing(self._extract(store, video_path, settings)) as extraction:
                    self._upscale(store, settings, extraction)
                self._check_stop()
                self._state(store, "encoding", error=None)
                self.task_status.start("encoding", "Encoding upscaled video")
                if source_frame_layout(store) == SOURCE_LAYOUT_SCALED:
                    size = source_frame_size(store)
                else:
                    original = decode_frame(store.get_frame("source", 0))
                    size = original.shape[1::-1]
                geometry = geometry_for_settings(size, settings)
                self.ffmpeg.encode(
                    video_path, output_path, store.get_metadata("fps"),
                    settings.codec, lambda: store.iter_frames("upscaled"), self.stop_event,
                    output_size=geometry.output_size, total_frames=store.get_metadata("total_frames"),
                    progress=self.task_status,
                )
                self._check_stop()
                self._state(store, "complete")
                self.task_status.finish("encoding")
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
            finally:
                self.task_status.clear()
        if not settings.keep_frames:
            cache_path.unlink()
            try:
                cache_path.parent.rmdir()
            except OSError:
                pass  # Never delete unrelated files in a cache directory.
        return str(output_path)


def upscale_video(
    process_status_q, event_stop_upscale_process, video_path, file_number,
    selected_output_path, selected_AI_model, selected_sharpening_amount,
    selected_AI_multithreading, selected_gpu, input_resize_factor,
    output_resize_factor, tiles_resolution, selected_video_extension,
    selected_video_codec, selected_keep_frames, *, clear_cache=False,
):
    """GUI adapter; backend modules never import the GUI entry point."""
    from .status import write_process_status

    settings = VideoSettings(
        model=selected_AI_model, gpu=selected_gpu,
        input_resize_factor=input_resize_factor, output_resize_factor=output_resize_factor,
        tiles_resolution=tiles_resolution, workers=selected_AI_multithreading,
        sharpening_amount=selected_sharpening_amount, codec=selected_video_codec,
        extension=selected_video_extension, keep_frames=selected_keep_frames,
    )
    directory = prepare_output_video_directory_name(
        video_path, selected_output_path, selected_AI_model,
        input_resize_factor, output_resize_factor, selected_sharpening_amount,
    )
    output = prepare_output_video_filename(
        video_path, selected_output_path, selected_AI_model,
        input_resize_factor, output_resize_factor, selected_video_extension,
        selected_sharpening_amount,
    )
    run_options = {"clear_cache": True} if clear_cache else {}
    return VideoUpscaler(
        stop_event=event_stop_upscale_process,
        progress=lambda snapshot: write_process_status(
            process_status_q, replace(
                snapshot, message=f"{file_number}. {snapshot.message}" if snapshot.message else "",
            )
        ),
    ).run(video_path, output, Path(directory) / "frames.sqlite3", settings, **run_options)


def get_resume_progress(video_path, selected_output_path, settings):
    """Read the same cache identity used by run(), without creating/changing it."""
    import sqlite3

    directory = prepare_output_video_directory_name(
        video_path, selected_output_path, settings.model, settings.input_resize_factor,
        settings.output_resize_factor, settings.sharpening_amount,
    )
    cache = Path(directory) / "frames.sqlite3"
    if not cache.is_file():
        return None
    try:
        source = Path(video_path).resolve()
        stat = source.stat()
        with FrameStore(cache, read_only=True) as store:
            if store.get_metadata("job_version") not in (2, CURRENT_JOB_VERSION):
                return None
            source_frame_layout(store)
            if store.get_metadata("source") != {
                "path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            } or store.get_metadata("processing_settings") != settings.processing_settings():
                return None
            total = store.get_metadata("total_frames")
            if not total or not store.get_metadata("extraction_complete"):
                return 0
            return min(100, int(store.frame_count("upscaled") * 100 / total))
    except (OSError, ValueError, sqlite3.Error):
        return None
