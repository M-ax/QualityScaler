import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest

from qualityscaler.frame_store import FrameStore


class FrameStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "résumé frames' #100%.sqlite3"

    def test_kinds_order_overwrite_and_binary_round_trip(self):
        with FrameStore(self.path) as store:
            self.assertEqual(store.frame_count("source"), 0)
            self.assertEqual(list(store.iter_frames("source")), [])
            for index in (10, 0, 4):
                store.put_frame("source", index, bytes([index, 0, 255]))
            store.put_frame("upscaled", 4, b"upscaled")
            store.put_frame("source", 4, bytearray(b"replaced"))
            store.put_frame("upscaled", 12, memoryview(b"buffer"))
            self.assertEqual(store.frame_count("source"), 3)
            self.assertEqual(store.frame_count("upscaled"), 2)
            self.assertEqual(list(store.iter_indices("source")), [0, 4, 10])
            self.assertEqual(
                list(store.iter_frames("source")),
                [b"\x00\x00\xff", b"replaced", b"\x0a\x00\xff"],
            )
            self.assertEqual(store.get_frame("upscaled", 4), b"upscaled")
            self.assertTrue(store.has_frame("source", 4))
            self.assertFalse(store.has_frame("upscaled", 0))
            with self.assertRaises(KeyError) as caught:
                store.get_frame("source", 99)
            self.assertEqual(caught.exception.args, (("source", 99),))
        with FrameStore(self.path, read_only=True) as reopened:
            self.assertEqual(reopened.get_frame("source", 4), b"replaced")
            self.assertEqual(reopened.get_frame("upscaled", 12), b"buffer")

    def test_metadata_round_trip_and_default(self):
        values = {
            "job": {"input": "é's video.mp4", "stage": "upscaling", "done": [0, 2]},
            "null": None,
            "boolean": False,
            "number": 1.25,
            "string": "ready",
        }
        with FrameStore(self.path) as store:
            sentinel = object()
            self.assertIs(store.get_metadata("absent", sentinel), sentinel)
            self.assertIsNone(store.get_metadata("absent"))
            for key, value in values.items():
                store.set_metadata(key, value)
                self.assertEqual(store.get_metadata(key), value)
            store.set_metadata("job", {"stage": "encoding"})
            values["job"] = {"stage": "encoding"}
        with FrameStore(self.path, read_only=True) as store:
            for key, value in values.items():
                self.assertEqual(store.get_metadata(key), value)

    def test_writes_and_metadata_are_not_committed_implicitly(self):
        with FrameStore(self.path) as writer:
            with FrameStore(self.path, read_only=True) as reader:
                writer.put_frame("source", 0, b"source")
                writer.set_metadata("extraction_complete", True)
                self.assertEqual(reader.frame_count("source"), 0)
                self.assertIsNone(reader.get_metadata("extraction_complete"))
                writer.commit()
                self.assertEqual(reader.frame_count("source"), 1)
                self.assertIs(reader.get_metadata("extraction_complete"), True)

    def test_close_commits_and_is_idempotent(self):
        store = FrameStore(self.path)
        store.put_frame("source", 0, b"frame")
        store.set_metadata("state", "done")
        store.close()
        store.close()
        with self.assertRaises(sqlite3.ProgrammingError):
            store.frame_count("source")
        with self.assertRaises(sqlite3.ProgrammingError):
            store.commit()
        with self.assertRaises(sqlite3.ProgrammingError):
            store.__enter__()
        with FrameStore(self.path, read_only=True) as reopened:
            self.assertEqual(reopened.get_frame("source", 0), b"frame")
            self.assertEqual(reopened.get_metadata("state"), "done")

    def test_exception_rolls_back_only_uncommitted_batch(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with FrameStore(self.path) as store:
                store.put_frame("source", 0, b"original")
                store.set_metadata("stage", "extracted")
                store.commit()
                store.put_frame("source", 0, b"replacement")
                store.put_frame("upscaled", 0, b"upscaled")
                store.set_metadata("stage", "upscaled")
                raise RuntimeError("interrupted")
        with FrameStore(self.path, read_only=True) as store:
            self.assertEqual(store.get_frame("source", 0), b"original")
            self.assertEqual(store.frame_count("upscaled"), 0)
            self.assertEqual(store.get_metadata("stage"), "extracted")

    def test_process_interruption_preserves_committed_resume_state(self):
        script = """
import os
import sys
from qualityscaler.frame_store import FrameStore
store = FrameStore(sys.argv[1])
store.put_frame("source", 0, b"source")
store.put_frame("upscaled", 0, b"finished")
store.set_metadata("state", {"stage": "upscaling", "last": 0})
store.commit()
store.put_frame("upscaled", 1, b"uncommitted")
store.set_metadata("state", {"stage": "upscaling", "last": 1})
os._exit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.path)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # Reopening writable also recovers any hot rollback journal.
        with FrameStore(self.path) as store:
            self.assertEqual(store.get_frame("source", 0), b"source")
            self.assertEqual(list(store.iter_indices("upscaled")), [0])
            self.assertEqual(
                store.get_metadata("state"), {"stage": "upscaling", "last": 0}
            )
            store.put_frame("upscaled", 1, b"resumed")
            store.set_metadata("state", {"stage": "encoding", "last": 1})
        with FrameStore(self.path, read_only=True) as store:
            self.assertEqual(store.get_frame("upscaled", 1), b"resumed")
            self.assertEqual(store.get_metadata("state")["stage"], "encoding")

    def test_reset_is_transactional_and_preserves_schema(self):
        with FrameStore(self.path) as store:
            store.put_frame("source", 0, b"source")
            store.put_frame("upscaled", 0, b"upscaled")
            store.set_metadata("job", "first")
        with self.assertRaises(RuntimeError):
            with FrameStore(self.path) as store:
                store.reset()
                self.assertEqual(store.frame_count("source"), 0)
                self.assertEqual(store.frame_count("upscaled"), 0)
                self.assertIsNone(store.get_metadata("job"))
                raise RuntimeError()
        with FrameStore(self.path) as store:
            self.assertEqual(store.frame_count("source"), 1)
            self.assertEqual(store.get_metadata("job"), "first")
            store.reset()
            store.commit()
        with FrameStore(self.path) as store:
            self.assertEqual(store.frame_count("source"), 0)
            self.assertEqual(store.frame_count("upscaled"), 0)
            self.assertIsNone(store.get_metadata("job"))
            store.put_frame("source", 8, b"new job")

    def test_iterators_do_not_keep_read_locks_across_yields(self):
        for iterator_name in ("iter_indices", "iter_frames"):
            with self.subTest(iterator=iterator_name):
                with FrameStore(self.path) as store:
                    store.reset()
                    for index in range(600):
                        store.put_frame("source", index, str(index).encode("ascii"))
                with FrameStore(self.path, read_only=True) as reader:
                    iterator = getattr(reader, iterator_name)("source")
                    first = next(iterator)
                    self.assertEqual(first, 0 if iterator_name == "iter_indices" else b"0")
                    writer = sqlite3.connect(self.path, timeout=0.05)
                    try:
                        writer.execute(
                            "INSERT INTO frames VALUES ('source', 700, ?)", (b"700",)
                        ).close()
                        writer.commit()  # Would fail with an active read cursor.
                    finally:
                        writer.close()
                    remaining = list(iterator)
                    if iterator_name == "iter_indices":
                        self.assertEqual(remaining, list(range(1, 600)) + [700])
                    else:
                        self.assertEqual(
                            remaining,
                            [str(index).encode("ascii") for index in range(1, 600)]
                            + [b"700"],
                        )

    def test_iteration_allows_writes_on_same_store(self):
        with FrameStore(self.path) as store:
            for index in (3, 0, 1):
                store.put_frame("source", index, str(index).encode("ascii"))
            store.commit()
            for index, frame in zip(
                store.iter_indices("source"), store.iter_frames("source")
            ):
                store.put_frame("upscaled", index, b"up " + frame)
                store.commit()
            self.assertEqual(
                list(store.iter_frames("upscaled")), [b"up 0", b"up 1", b"up 3"]
            )

    def test_thread_uses_an_independent_connection(self):
        with FrameStore(self.path) as writer:
            writer.put_frame("source", 0, b"committed")
            writer.commit()
            result = []

            def read():
                try:
                    with FrameStore(self.path, read_only=True) as reader:
                        result.append(reader.get_frame("source", 0))
                except BaseException as error:
                    result.append(error)

            thread = threading.Thread(target=read)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, [b"committed"])

    def test_read_only_cannot_write_or_create_missing_file(self):
        with self.assertRaises(sqlite3.OperationalError):
            FrameStore(self.path, read_only=True)
        self.assertFalse(self.path.exists())
        with FrameStore(self.path) as store:
            store.put_frame("source", 0, b"frame")
        with FrameStore(self.path, read_only=True) as reader:
            for write in (
                lambda: reader.put_frame("source", 0, b"changed"),
                lambda: reader.set_metadata("job", 1),
                reader.reset,
            ):
                with self.assertRaisesRegex(sqlite3.OperationalError, "read-only"):
                    write()
            reader.commit()  # A read-only commit is a harmless no-op.
            self.assertEqual(reader.get_frame("source", 0), b"frame")

    def test_validates_kind_indices_data_and_metadata(self):
        with FrameStore(self.path) as store:
            for kind in ("", "SOURCE", "other", None, 1):
                for operation in (
                    lambda: store.put_frame(kind, 0, b"x"),
                    lambda: store.get_frame(kind, 0),
                    lambda: store.has_frame(kind, 0),
                    lambda: store.frame_count(kind),
                    lambda: list(store.iter_indices(kind)),
                    lambda: list(store.iter_frames(kind)),
                ):
                    with self.subTest(kind=kind), self.assertRaises(ValueError):
                        operation()
            for index, exception in (
                (-1, ValueError),
                (1.5, TypeError),
                ("0", TypeError),
                (True, TypeError),
                (None, TypeError),
                (2**63, OverflowError),
            ):
                for operation in (
                    lambda: store.put_frame("source", index, b"x"),
                    lambda: store.get_frame("source", index),
                    lambda: store.has_frame("source", index),
                ):
                    with self.subTest(index=index), self.assertRaises(exception):
                        operation()
            store.put_frame("source", 2**63 - 1, b"last possible")
            self.assertEqual(list(store.iter_indices("source")), [2**63 - 1])
            for data in ("text", 12, None):
                with self.subTest(data=data), self.assertRaises(TypeError):
                    store.put_frame("source", 0, data)
            for key in (None, 1, b"key"):
                with self.subTest(key=key), self.assertRaises(TypeError):
                    store.set_metadata(key, 1)
                with self.assertRaises(TypeError):
                    store.get_metadata(key)
            for value, exception in (
                (object(), TypeError),
                (float("nan"), ValueError),
                (float("inf"), ValueError),
            ):
                with self.assertRaises(exception):
                    store.set_metadata("invalid", value)
            self.assertIsNone(store.get_metadata("invalid"))

    def test_rollback_journal_and_schema_version(self):
        with FrameStore(self.path) as store:
            store.put_frame("source", 0, b"one")
            store.commit()
        connection = sqlite3.connect(self.path)
        try:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "delete")
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
        finally:
            connection.close()
        self.assertFalse(Path(str(self.path) + "-wal").exists())
        self.assertFalse(Path(str(self.path) + "-shm").exists())
        self.assertFalse(Path(str(self.path) + "-journal").exists())

    def test_rejects_unknown_versions_and_closes_failed_constructor(self):
        with FrameStore(self.path):
            pass
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA user_version = 999")
            connection.commit()
        finally:
            connection.close()
        for read_only in (False, True):
            with self.subTest(read_only=read_only):
                with self.assertRaisesRegex(ValueError, "schema version 999"):
                    FrameStore(self.path, read_only=read_only)
        # Windows also verifies that failed constructors did not leak handles.
        self.path.unlink()
        with FrameStore(self.path):
            pass

    def test_rejects_unrelated_and_incomplete_schemas(self):
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(ValueError, "not an initialized FrameStore"):
            FrameStore(self.path)
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA user_version = 1")
            connection.commit()
        finally:
            connection.close()
        for read_only in (False, True):
            with self.assertRaisesRegex(ValueError, "invalid FrameStore schema"):
                FrameStore(self.path, read_only=read_only)

    def test_rejects_corrupt_database_and_corrupt_stored_values(self):
        self.path.write_bytes(b"this is not a SQLite database" * 10)
        for read_only in (False, True):
            with self.assertRaises(sqlite3.DatabaseError):
                FrameStore(self.path, read_only=read_only)
        self.path.unlink()
        with FrameStore(self.path) as store:
            store.put_frame("source", 0, b"frame")
            store.set_metadata("state", "valid")
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA ignore_check_constraints = ON")
            connection.execute("UPDATE frames SET data = 'not a BLOB'")
            connection.execute("UPDATE metadata SET value = 'broken JSON'")
            connection.commit()
        finally:
            connection.close()
        with FrameStore(self.path, read_only=True) as store:
            with self.assertRaisesRegex(ValueError, "not a BLOB"):
                store.get_frame("source", 0)
            with self.assertRaisesRegex(ValueError, "not a BLOB"):
                list(store.iter_frames("source"))
            with self.assertRaises(json.JSONDecodeError):
                store.get_metadata("state")

    def test_read_only_rejects_uninitialized_database(self):
        connection = sqlite3.connect(self.path)
        connection.close()
        with self.assertRaisesRegex(ValueError, "schema version 0"):
            FrameStore(self.path, read_only=True)


if __name__ == "__main__":
    unittest.main()
