"""Transactional, single-file storage for source and upscaled JPEG frames.

Each ``FrameStore`` owns its SQLite connection. Create a separate instance in
each thread/process; do not pass an open store to another thread or a child
process. JPEG data is stored as an opaque BLOB, without importing an image
decoder.

Writes, including metadata updates and ``reset()``, remain in one transaction
until ``commit()``. Explicit ``close()`` and a successful context-manager exit
commit pending writes. An exceptional context-manager exit instead rolls back
pending writes and closes the connection; previously committed batches survive.
SQLite uses a DELETE rollback journal and FULL synchronization, not WAL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from collections.abc import Iterator
from typing import Any


_SCHEMA_VERSION = 1
_KINDS = ("source", "upscaled")
_INDEX_PAGE_SIZE = 256


class FrameStore:
    """Store JPEG BLOBs by ``(kind, frame_index)`` in a SQLite file.

    ``kind`` is ``"source"`` or ``"upscaled"``. Indices are nonnegative Python
    integers representable by SQLite (booleans are not indices). Metadata keys
    are strings and values must be JSON serializable, without NaN or infinity.

    Iterators use short, keyset-ordered queries and close their cursors before
    yielding. They do not hold a read snapshot/lock across yields, or materialize
    all frames/indices. Concurrent changes ahead of their current position may
    be observed; use a completed kind for a stable traversal.
    """

    def __init__(
        self, path: str | os.PathLike[str], read_only: bool = False
    ) -> None:
        self._connection: sqlite3.Connection | None = None
        self._read_only = read_only
        file_path = Path(path).resolve()

        if read_only:
            uri = file_path.as_uri()
            # SQLite normally rejects a URI authority other than localhost.
            # Represent a Windows UNC path as a path, not a URI authority.
            if uri.startswith("file://") and not uri.startswith("file:///"):
                uri = "file:////" + uri[len("file://") :]
            target = uri + "?mode=ro"
        else:
            target = str(file_path)

        try:
            self._connection = sqlite3.connect(
                target, uri=read_only, isolation_level="DEFERRED"
            )
            self._prepare_schema()
        except BaseException:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            raise

    def __enter__(self) -> FrameStore:
        self._require_connection()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._close(commit=exc_type is None)

    def close(self) -> None:
        """Commit pending writes and close; calling this twice is harmless.

        To roll back on a processing error, use the store as a context manager
        and allow the exception to leave the ``with`` block.
        """
        self._close(commit=True)

    def _close(self, *, commit: bool) -> None:
        connection = self._connection
        if connection is None:
            return
        self._connection = None
        try:
            if commit:
                connection.commit()
            else:
                connection.rollback()
        finally:
            connection.close()

    def commit(self) -> None:
        """Atomically publish pending frames and resume metadata."""
        self._require_connection().commit()

    def put_frame(self, kind: str, index: int, data: bytes) -> None:
        """Insert or replace a frame, without committing the transaction."""
        self._validate_kind(kind)
        self._validate_index(index)
        connection = self._writable_connection()
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("frame data must be bytes-like")
        connection.execute(
            "INSERT INTO frames (kind, frame_index, data) VALUES (?, ?, ?) "
            "ON CONFLICT (kind, frame_index) DO UPDATE SET data = excluded.data",
            (kind, index, sqlite3.Binary(bytes(data))),
        ).close()

    def get_frame(self, kind: str, index: int) -> bytes:
        """Return a frame, or raise ``KeyError((kind, index))`` if absent."""
        self._validate_kind(kind)
        self._validate_index(index)
        row = self._query_one(
            "SELECT data FROM frames WHERE kind = ? AND frame_index = ?",
            (kind, index),
        )
        if row is None:
            raise KeyError((kind, index))
        return self._blob(row[0])

    def has_frame(self, kind: str, index: int) -> bool:
        self._validate_kind(kind)
        self._validate_index(index)
        return self._query_one(
            "SELECT 1 FROM frames WHERE kind = ? AND frame_index = ?",
            (kind, index),
        ) is not None

    def frame_count(self, kind: str) -> int:
        self._validate_kind(kind)
        return self._query_one(
            "SELECT COUNT(*) FROM frames WHERE kind = ?", (kind,)
        )[0]

    def iter_indices(self, kind: str) -> Iterator[int]:
        """Yield ascending indices in bounded pages, without a live cursor."""
        self._validate_kind(kind)
        previous = -1
        while True:
            cursor = self._require_connection().execute(
                "SELECT frame_index FROM frames "
                "WHERE kind = ? AND frame_index > ? "
                "ORDER BY frame_index LIMIT ?",
                (kind, previous, _INDEX_PAGE_SIZE),
            )
            try:
                rows = cursor.fetchall()
            finally:
                cursor.close()
            if not rows:
                return
            for (index,) in rows:
                previous = index
                yield index

    def iter_frames(self, kind: str) -> Iterator[bytes]:
        """Yield ascending JPEG BLOBs, loading only one frame per query."""
        self._validate_kind(kind)
        previous = -1
        while True:
            row = self._query_one(
                "SELECT frame_index, data FROM frames "
                "WHERE kind = ? AND frame_index > ? "
                "ORDER BY frame_index LIMIT 1",
                (kind, previous),
            )
            if row is None:
                return
            previous = row[0]
            yield self._blob(row[1])

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Read JSON metadata; malformed stored JSON raises ``ValueError``."""
        self._validate_metadata_key(key)
        row = self._query_one("SELECT value FROM metadata WHERE key = ?", (key,))
        return default if row is None else json.loads(row[0])

    def set_metadata(self, key: str, value: Any) -> None:
        """Set JSON metadata in the same transaction as pending frames."""
        self._validate_metadata_key(key)
        connection = self._writable_connection()
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, encoded),
        ).close()

    def reset(self) -> None:
        """Delete all frames and metadata, without committing."""
        connection = self._writable_connection()
        connection.execute("DELETE FROM frames").close()
        connection.execute("DELETE FROM metadata").close()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise sqlite3.ProgrammingError("FrameStore is closed")
        return self._connection

    def _writable_connection(self) -> sqlite3.Connection:
        connection = self._require_connection()
        if self._read_only:
            raise sqlite3.OperationalError("FrameStore is read-only")
        return connection

    def _query_one(
        self, sql: str, parameters: tuple[Any, ...] = ()
    ) -> tuple[Any, ...] | None:
        cursor = self._require_connection().execute(sql, parameters)
        try:
            return cursor.fetchone()
        finally:
            cursor.close()

    @staticmethod
    def _validate_kind(kind: str) -> None:
        if kind not in _KINDS:
            raise ValueError("kind must be 'source' or 'upscaled'")

    @staticmethod
    def _validate_index(index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("frame index must be an integer")
        if index < 0:
            raise ValueError("frame index must be nonnegative")
        if index > 2**63 - 1:
            raise OverflowError("frame index exceeds SQLite's integer range")

    @staticmethod
    def _validate_metadata_key(key: str) -> None:
        if not isinstance(key, str):
            raise TypeError("metadata key must be a string")

    @staticmethod
    def _blob(data: Any) -> bytes:
        if not isinstance(data, bytes):
            raise ValueError("stored frame data is not a BLOB")
        return data

    def _check_version(self, version: int) -> None:
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FrameStore schema version {version}; "
                f"expected {_SCHEMA_VERSION}"
            )

    def _check_empty_database(self) -> None:
        if self._query_one(
            "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1"
        ):
            raise ValueError("database is not an initialized FrameStore")

    def _validate_tables(self) -> None:
        expected = {
            "frames": [
                ("kind", "TEXT", 1, 1),
                ("frame_index", "INTEGER", 1, 2),
                ("data", "BLOB", 1, 0),
            ],
            "metadata": [("key", "TEXT", 1, 1), ("value", "TEXT", 1, 0)],
        }
        for table, columns in expected.items():
            cursor = self._require_connection().execute(f"PRAGMA table_info({table})")
            try:
                actual = [
                    (row[1], row[2], row[3], row[5]) for row in cursor.fetchall()
                ]
            finally:
                cursor.close()
            if actual != columns:
                raise ValueError(f"invalid FrameStore schema for table {table!r}")

    def _prepare_schema(self) -> None:
        version = self._query_one("PRAGMA user_version")[0]
        if version == 0 and not self._read_only:
            self._check_empty_database()
        else:
            self._check_version(version)
            self._validate_tables()

        if self._read_only:
            if self._query_one("PRAGMA journal_mode")[0].lower() == "wal":
                raise ValueError("FrameStore requires rollback journaling, not WAL")
            return

        connection = self._require_connection()
        mode = self._query_one("PRAGMA journal_mode = DELETE")[0]
        if mode.lower() != "delete":
            raise ValueError("could not enable SQLite DELETE rollback journaling")
        connection.execute("PRAGMA synchronous = FULL").close()

        if version == 0:
            # Recheck under the write lock in case another opener initialized
            # the empty file between the first check and this transaction.
            connection.execute("BEGIN IMMEDIATE").close()
            version = self._query_one("PRAGMA user_version")[0]
            if version == 0:
                self._check_empty_database()
                connection.execute(
                    "CREATE TABLE frames ("
                    "kind TEXT NOT NULL CHECK (kind IN ('source', 'upscaled')), "
                    "frame_index INTEGER NOT NULL CHECK (frame_index >= 0), "
                    "data BLOB NOT NULL CHECK (typeof(data) = 'blob'), "
                    "PRIMARY KEY (kind, frame_index)) WITHOUT ROWID"
                ).close()
                connection.execute(
                    "CREATE TABLE metadata ("
                    "key TEXT NOT NULL PRIMARY KEY, value TEXT NOT NULL"
                    ") WITHOUT ROWID"
                ).close()
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}").close()
            else:
                self._check_version(version)
            self._validate_tables()
            connection.commit()
