"""Latest status per active task, published together without competing writers."""

from dataclasses import dataclass
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


@dataclass
class _StageProgress:
    message: str
    completed: int
    initial_completed: int
    total: int | None
    started: float
    estimated_total: bool

    def remaining(self, now):
        if self.total is None or self.total <= 0:
            return None
        if self.completed >= self.total:
            # A probe can undercount. Reaching its estimate is not evidence
            # that extraction or inference has no work left before EOF.
            return None if self.estimated_total else 0
        processed = self.completed - self.initial_completed
        elapsed = now - self.started
        if processed <= 0 or elapsed <= 0:
            return None
        # Only this run's work contributes to the rate, not resumed cache hits.
        return elapsed / processed * (self.total - self.completed)


class PipelineStatus:
    """Keep each active stage's latest message and its own timing estimate.

    Stages report absolute counts, never preformatted timing strings. Concurrent
    stages occupy separate lines; their estimates are never combined. Unknown
    totals/rates affect only that stage. Unstarted stages and later files are
    deliberately not included in a stage's estimate.
    """

    def __init__(self, publish, *, clock=None, now=None):
        self._publish = publish
        self._clock = clock if clock is not None else monotonic
        self._now = now if now is not None else lambda: datetime.now(timezone.utc)
        self._stages = {}
        self._lock = Lock()

    def start(self, stage, message, *, completed=0, total=None, estimated_total=False):
        """Start a stage, or reset its rate when an encoder attempt is retried."""
        with self._lock:
            self._stages[stage] = _StageProgress(
                message, completed, completed, total, self._clock(), estimated_total,
            )
            self._emit()

    def update(self, stage, message=None, *, completed=None, total=None, estimated_total=None):
        with self._lock:
            progress = self._stages.get(stage)
            if progress is None:
                return  # A late callback must not resurrect a finished stage.
            if message is not None:
                progress.message = message
            if completed is not None:
                progress.completed = max(progress.completed, completed)
            if total is not None:
                progress.total = total
            if estimated_total is not None:
                progress.estimated_total = estimated_total
            self._emit()

    def finish(self, stage):
        with self._lock:
            if self._stages.pop(stage, None) is not None:
                self._emit()

    def clear(self):
        """Discard interrupted tasks when the owner stops or fails."""
        with self._lock:
            if self._stages:
                self._stages.clear()
                self._emit()

    def _emit(self):
        # Hold the lock through publication: otherwise an older snapshot could
        # overwrite a newer one in the GUI's single-slot queue.
        now = self._clock()
        wall_now = self._now()
        lines = []
        for stage in self._stages.values():
            remaining = stage.remaining(now)
            if remaining is None:
                timing = "Remaining: estimating... | ETA (local): estimating..."
            else:
                seconds = max(0, ceil(remaining))
                # Add elapsed seconds in UTC, then convert using the system's
                # rules at arrival (including daylight-saving transitions).
                eta = (wall_now + timedelta(seconds=seconds)).astimezone()
                timing = (
                    f"Remaining: {format_duration(seconds)}"
                    f" | ETA (local): {eta:%Y-%m-%d %H:%M:%S %Z}"
                )
            lines.append(f"{stage.message} | {timing}")
        self._publish("\n".join(lines))


@dataclass(frozen=True)
class VideoProgress:
    """One replaceable snapshot of both video stages, safe to send across processes.

    The total is an estimate until extraction completes, or None if unknown.
    Counts include resumed frames and advance only after writing to the cache.
    """

    message: str
    extracted: int
    upscaled: int
    total: int | None
    extraction_complete: bool

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
