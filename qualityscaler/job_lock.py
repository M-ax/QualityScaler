"""Crash-released, same-machine exclusion for an entire QualityScaler job.

The lock lives in the OS temporary directory, not beside the SQLite cache.
This leaves the cache directory unchanged and allows cache cleanup while the
job is still locked. A SQLite transaction is not suitable: a job spans many
committed batches, multiple connections, and eventual database removal.

Lock files deliberately persist after release. Unlinking a lock file can let
another process lock a new inode while an existing opener still uses the old
one. Only the open handle's OS lock represents ownership; file existence does
not. The OS releases ownership even after a crash or forced termination.
"""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, BinaryIO

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class JobAlreadyRunning(RuntimeError):
    """Another handle already owns the lock for this cache path."""


def _lock_file(file: BinaryIO) -> None:
    file.seek(0)
    if os.name == "nt":
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(file: BinaryIO) -> None:
    file.seek(0)
    if os.name == "nt":
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


class JobLock:
    """Nonblocking context manager keyed by a normalized absolute cache path.

    Acquire this before preparing/opening the database and hold it through
    final cache cleanup. A competing context raises ``JobAlreadyRunning``;
    unrelated filesystem errors propagate normally. The cache/database itself
    is never created, opened, or modified by this class.

    Instances can be reused after release, but are not reentrant. The lock
    file contains only one byte, no PID or stale ownership metadata.
    """

    def __init__(self, cache_path: str | os.PathLike[str]) -> None:
        self._cache_path = Path(cache_path).resolve()
        normalized = os.path.normcase(str(self._cache_path))
        digest = hashlib.sha256(os.fsencode(normalized)).hexdigest()
        self._lock_path = (
            Path(tempfile.gettempdir()) / "qualityscaler-locks" / f"{digest}.lock"
        )
        self._file: BinaryIO | None = None

    def __enter__(self) -> JobLock:
        if self._file is not None:
            raise self._already_running()

        directory = self._lock_path.parent
        directory.mkdir(mode=0o700, exist_ok=True)
        try:
            os.chmod(directory, 0o700)
        except OSError:
            # Existing directories/filesystems may not permit chmod, and
            # Windows does not implement POSIX permission bits fully.
            pass

        descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            file = os.fdopen(descriptor, "r+b", buffering=0)
        except BaseException:
            os.close(descriptor)
            raise

        try:
            if os.fstat(file.fileno()).st_size == 0:
                file.seek(0)
                if file.write(b"\x00") != 1:
                    raise OSError(errno.EIO, "could not initialize job lock file")
                file.flush()
            _lock_file(file)
        except BaseException as error:
            file.close()
            if isinstance(error, OSError) and error.errno in (
                errno.EACCES,
                errno.EAGAIN,
                errno.EDEADLK,
            ):
                # On Windows a competing first-time opener can lock byte zero
                # between fstat() and write(). Its mandatory lock can reject
                # that initialization write, not just the locking() call.
                raise self._already_running() from error
            raise

        self._file = file
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release ownership and close the handle, without unlinking the file."""
        file = self._file
        if file is None:
            return
        self._file = None
        try:
            _unlock_file(file)
        finally:
            file.close()

    def _already_running(self) -> JobAlreadyRunning:
        return JobAlreadyRunning(
            f"QualityScaler job is already running for cache {self._cache_path}"
        )
