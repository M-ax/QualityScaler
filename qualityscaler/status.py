"""Immutable per-stage snapshots, published together without competing writers."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from math import ceil
from queue import Empty
from threading import Lock
from time import monotonic


def format_duration(seconds):
    """Round up estimates and keep day/hour/minute/second boundaries readable."""
    seconds = max(0, ceil(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = []
    for value, unit in ((days, "d"), (hours, "h"), (minutes, "m"), (seconds, "s")):
        if value or parts or unit == "s":
            parts.append(f"{value}{unit}")
    return " ".join(parts)


@dataclass(frozen=True)
class StageStatus:
    """One persistent row; only explicit success gives it the complete state."""

    stage: str
    message: str = ""
    state: str = "waiting"
    completed: int = 0
    total: int | None = None
    estimated_total: bool = False
    remaining_seconds: int | None = None
    eta: str | None = None


@dataclass
class _StageProgress:
    status: StageStatus
    initial_completed: int
    started: float

    def remaining(self, now):
        status = self.status
        if status.total is None or status.total <= 0:
            return None
        if status.completed >= status.total:
            # A probe can undercount. Reaching its estimate is not evidence
            # that extraction or inference has no work left before EOF.
            return None if status.estimated_total else 0
        processed = status.completed - self.initial_completed
        elapsed = now - self.started
        if processed <= 0 or elapsed <= 0:
            return None
        # Only this run's work contributes to the rate, not resumed cache hits.
        return elapsed / processed * (status.total - status.completed)


class PipelineStatus:
    """Keep each stage's latest message, explicit state and own timing estimate.

    Stages report absolute counts, never preformatted timing strings. Concurrent
    stages occupy separate lines; their estimates are never combined. Unknown
    totals/rates affect only that stage. Unstarted stages and later files are
    deliberately not included in a stage's estimate. Completed/interrupted rows
    retain their final timing until clear() or a new start() for that stage.

    ``stages`` is an immutable tuple, assigned before the string callback runs.
    A callback may read it directly without re-entering this reporter's lock.
    """

    STAGES = ("extracting", "upscaling", "encoding")

    def __init__(self, publish, *, clock=None, now=None):
        self._publish = publish
        self._clock = clock if clock is not None else monotonic
        self._now = now if now is not None else lambda: datetime.now(timezone.utc)
        self.stages = tuple(StageStatus(stage) for stage in self.STAGES)
        self._stages = {
            status.stage: _StageProgress(status, 0, 0) for status in self.stages
        }
        self._lock = Lock()

    def start(self, stage, message, *, completed=0, total=None, estimated_total=False):
        """Start a stage, or reset its rate when an encoder attempt is retried."""
        with self._lock:
            self._stages[stage] = _StageProgress(
                StageStatus(stage, message, "running", completed, total, estimated_total),
                completed, self._clock(),
            )
            self._emit()

    def update(self, stage, message=None, *, completed=None, total=None, estimated_total=None):
        with self._lock:
            progress = self._stages.get(stage)
            if progress is None or progress.status.state != "running":
                return  # A late callback must not resurrect a finished stage.
            status = progress.status
            progress.status = replace(
                status,
                message=status.message if message is None else message,
                completed=status.completed if completed is None else max(status.completed, completed),
                total=status.total if total is None else total,
                estimated_total=status.estimated_total if estimated_total is None else estimated_total,
            )
            self._emit()

    def finish(self, stage, message=None, *, completed=None, total=None):
        """Record verified success, including validated stages restored from cache.

        Counts alone never imply success. Call only after saving, validation and
        resource cleanup (or, for encoding, successful output replacement).
        """
        with self._lock:
            progress = self._stages.get(stage)
            if progress is None or progress.status.state in ("complete", "stopped", "error"):
                return
            status = progress.status
            progress.status = replace(
                status, state="complete",
                message=status.message if message is None else message,
                completed=status.completed if completed is None else completed,
                total=status.total if total is None else total,
                estimated_total=False, remaining_seconds=0, eta=self._local_eta(self._now(), 0),
            )
            self._emit()

    def interrupt(self, state="stopped", *, stage=None):
        """Freeze running rows on cancellation/failure, preserving successful rows."""
        if state not in ("stopped", "error"):
            raise ValueError("Interrupted state must be stopped or error")
        with self._lock:
            changed = False
            for key, progress in self._stages.items():
                if (stage is None or key == stage) and progress.status.state == "running":
                    progress.status = replace(progress.status, state=state)
                    changed = True
            if changed:
                self._emit()

    def clear(self):
        """Reset all rows for a new job; not an interruption or success signal."""
        with self._lock:
            if self.stages == tuple(StageStatus(stage) for stage in self.STAGES):
                return
            self._stages = {
                stage: _StageProgress(StageStatus(stage), 0, 0) for stage in self.STAGES
            }
            self._emit()

    @staticmethod
    def _local_eta(wall_now, seconds):
        # Add elapsed seconds in UTC, then convert using the system's rules at
        # arrival (including daylight-saving transitions).
        eta = (wall_now + timedelta(seconds=seconds)).astimezone()
        return f"{eta:%Y-%m-%d %H:%M:%S %Z}"

    def _emit(self):
        # Hold the lock through publication: otherwise an older snapshot could
        # overwrite a newer one in the GUI's single-slot queue.
        now = self._clock()
        wall_now = self._now()
        lines = []
        for progress in self._stages.values():
            status = progress.status
            if status.state == "running":
                remaining = progress.remaining(now)
                seconds = None if remaining is None else max(0, ceil(remaining))
                status = progress.status = replace(
                    status, remaining_seconds=seconds,
                    eta=None if seconds is None else self._local_eta(wall_now, seconds),
                )
            if status.state == "waiting":
                continue
            if status.remaining_seconds is None:
                timing = "Remaining: estimating... | ETA (local): estimating..."
            else:
                timing = (
                    f"Remaining: {format_duration(status.remaining_seconds)}"
                    f" | ETA (local): {status.eta}"
                )
            state = "" if status.state == "running" else f" | {status.state.capitalize()}"
            lines.append(f"{status.message} | {timing}{state}")
        self.stages = tuple(progress.status for progress in self._stages.values())
        self._publish("\n".join(lines))


@dataclass(frozen=True)
class VideoProgress:
    """One replaceable snapshot of video stages, safe to send across processes.

    The total is an estimate until extraction completes, or None if unknown.
    Counts include resumed frames and advance only after writing to the cache.
    """

    message: str
    extracted: int
    upscaled: int
    total: int | None
    extraction_complete: bool
    stages: tuple[StageStatus, ...] = ()

    def __str__(self):
        return self.message


@dataclass(frozen=True)
class CacheSettingsConflict:
    """A terminal conflict that can be retried only after explicit user consent."""

    message: str
    file_number: int

    def __str__(self):
        return self.message


def write_process_status(process_status_q, step):
    # empty() is only a hint on multiprocessing queues; never follow it with a
    # blocking get, which could hang the orchestrator during shutdown.
    while True:
        try:
            process_status_q.get_nowait()
        except Empty:
            break
    process_status_q.put(
        step if isinstance(step, (VideoProgress, CacheSettingsConflict)) else str(step)
    )
