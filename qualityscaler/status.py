"""Latest-status queue shared by the GUI and backend."""

from dataclasses import dataclass
from queue import Empty


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
