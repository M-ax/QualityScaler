"""Exercise actual OS locks, including subprocess interruption."""

import errno
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from qualityscaler.job_lock import JobAlreadyRunning, JobLock


class JobLockTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.os_temp = self.root / "OS temporary files"
        self.os_temp.mkdir()
        self.cache = self.root / "output cache" / "résumé's frames.sqlite3"
        patcher = mock.patch(
            "qualityscaler.job_lock.tempfile.gettempdir",
            return_value=str(self.os_temp),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def expected_lock_path(self, cache=None):
        cache = self.cache if cache is None else cache
        normalized = os.path.normcase(str(Path(cache).resolve()))
        digest = hashlib.sha256(os.fsencode(normalized)).hexdigest()
        return self.os_temp / "qualityscaler-locks" / f"{digest}.lock"

    def run_competitor(self, cache=None):
        script = """
import sys
import tempfile
from qualityscaler.job_lock import JobAlreadyRunning, JobLock
tempfile.tempdir = sys.argv[2]
try:
    with JobLock(sys.argv[1]):
        print("acquired")
except JobAlreadyRunning as error:
    print(str(error))
    sys.exit(23)
"""
        return subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(self.cache if cache is None else cache),
                str(self.os_temp),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def start_owner(self):
        script = """
import os
import sys
import tempfile
from qualityscaler.job_lock import JobLock
tempfile.tempdir = sys.argv[2]
with JobLock(sys.argv[1]):
    print("locked", flush=True)
    if sys.stdin.readline().strip() == "exit":
        os._exit(0)
"""
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(self.cache),
                str(self.os_temp),
            ],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def cleanup():
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)

        self.addCleanup(cleanup)
        ready = process.stdout.readline()
        if ready != "locked\n":
            output, errors = process.communicate(timeout=10)
            self.fail(f"owner did not acquire lock: {ready}{output}{errors}")
        return process

    def test_same_process_separate_handles_contend(self):
        with JobLock(self.cache):
            with self.assertRaisesRegex(JobAlreadyRunning, "already running"):
                with JobLock(self.cache):
                    self.fail("competing context entered")
        with JobLock(self.cache):
            pass

    def test_subprocess_cannot_acquire_held_lock(self):
        with JobLock(self.cache):
            competitor = self.run_competitor()
            self.assertEqual(competitor.returncode, 23, competitor.stderr)
            self.assertIn("already running", competitor.stdout)
        competitor = self.run_competitor()
        self.assertEqual(competitor.returncode, 0, competitor.stderr)
        self.assertEqual(competitor.stdout.strip(), "acquired")

    def test_abrupt_process_exit_releases_lock(self):
        owner = self.start_owner()
        with self.assertRaises(JobAlreadyRunning):
            with JobLock(self.cache):
                pass
        owner.stdin.write("exit\n")
        owner.stdin.flush()
        output, errors = owner.communicate(timeout=10)
        self.assertEqual(owner.returncode, 0, output + errors)
        with JobLock(self.cache):
            pass
        self.assertTrue(self.expected_lock_path().exists())

    def test_killed_process_releases_lock(self):
        owner = self.start_owner()
        with self.assertRaises(JobAlreadyRunning):
            with JobLock(self.cache):
                pass
        owner.kill()
        owner.communicate(timeout=10)
        with JobLock(self.cache):
            pass

    def test_different_cache_paths_are_independent(self):
        other_cache = self.cache.with_name("other.sqlite3")
        with JobLock(self.cache):
            with JobLock(other_cache):
                with self.assertRaises(JobAlreadyRunning):
                    with JobLock(other_cache):
                        pass
                competitor = self.run_competitor(
                    self.cache.with_name("third.sqlite3")
                )
                self.assertEqual(competitor.returncode, 0, competitor.stderr)
        self.assertNotEqual(
            self.expected_lock_path(), self.expected_lock_path(other_cache)
        )

    def test_normalized_paths_share_lock(self):
        alias = self.cache.parent / "unused" / ".." / self.cache.name
        self.assertEqual(self.expected_lock_path(), self.expected_lock_path(alias))
        with JobLock(self.cache):
            with self.assertRaises(JobAlreadyRunning):
                with JobLock(alias):
                    pass
            if os.name == "nt":
                with self.assertRaises(JobAlreadyRunning):
                    with JobLock(str(self.cache).upper()):
                        pass

    def test_persistent_tiny_temp_file_without_cache_directory(self):
        lock = JobLock(self.cache)
        path = self.expected_lock_path()
        self.assertFalse(path.parent.exists())
        self.assertFalse(self.cache.parent.exists())
        with lock:
            before = path.stat()
            self.assertEqual(before.st_size, 1)
            self.assertFalse(self.cache.parent.exists())
        self.assertTrue(path.exists())
        self.assertEqual(path.read_bytes(), b"\x00")
        self.assertEqual(path.stat().st_ino, before.st_ino)
        self.assertEqual(path.stat().st_size, 1)
        with JobLock(self.cache):
            self.assertEqual(path.stat().st_ino, before.st_ino)
        self.assertFalse(self.cache.parent.exists())
        self.assertEqual(list(path.parent.iterdir()), [path])
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)

    def test_can_remove_cache_while_locked(self):
        self.cache.parent.mkdir()
        self.cache.write_bytes(b"cache")
        with JobLock(self.cache):
            self.assertEqual(self.cache.read_bytes(), b"cache")
            self.cache.unlink()
            self.cache.parent.rmdir()
            with self.assertRaises(JobAlreadyRunning):
                with JobLock(self.cache):
                    pass
        self.assertFalse(self.cache.parent.exists())

    def test_body_exception_releases_lock(self):
        with self.assertRaisesRegex(RuntimeError, "job failed"):
            with JobLock(self.cache):
                raise RuntimeError("job failed")
        with JobLock(self.cache):
            pass

    def test_instance_reuse_nonreentrancy_and_idempotent_close(self):
        lock = JobLock(self.cache)
        lock.close()
        with lock as entered:
            self.assertIs(entered, lock)
            with self.assertRaises(JobAlreadyRunning):
                lock.__enter__()
            with self.assertRaises(JobAlreadyRunning):
                with JobLock(self.cache):
                    pass
        lock.close()
        with lock:
            pass

    def test_best_effort_directory_permissions(self):
        with mock.patch(
            "qualityscaler.job_lock.os.chmod",
            side_effect=PermissionError("chmod unavailable"),
        ):
            with JobLock(self.cache):
                pass

    def test_failed_contention_closes_its_file_handle(self):
        handles = []
        real_fdopen = os.fdopen

        def fdopen(*args, **kwargs):
            file = real_fdopen(*args, **kwargs)
            handles.append(file)
            return file

        with JobLock(self.cache):
            with mock.patch("qualityscaler.job_lock.os.fdopen", side_effect=fdopen):
                with self.assertRaises(JobAlreadyRunning):
                    with JobLock(self.cache):
                        pass
            self.assertEqual(len(handles), 1)
            self.assertTrue(handles[0].closed)

    def test_unexpected_acquisition_error_closes_handle_and_propagates(self):
        handles = []
        real_fdopen = os.fdopen

        def fdopen(*args, **kwargs):
            file = real_fdopen(*args, **kwargs)
            handles.append(file)
            return file

        with mock.patch("qualityscaler.job_lock.os.fdopen", side_effect=fdopen):
            with mock.patch(
                "qualityscaler.job_lock._lock_file",
                side_effect=OSError(errno.EIO, "device error"),
            ):
                with self.assertRaises(OSError) as caught:
                    with JobLock(self.cache):
                        pass
        self.assertEqual(caught.exception.errno, errno.EIO)
        self.assertEqual(len(handles), 1)
        self.assertTrue(handles[0].closed)
        with JobLock(self.cache):
            pass

    def test_fdopen_failure_closes_raw_descriptor(self):
        descriptors = []
        real_open = os.open

        def open_descriptor(*args, **kwargs):
            descriptor = real_open(*args, **kwargs)
            descriptors.append(descriptor)
            return descriptor

        with mock.patch(
            "qualityscaler.job_lock.os.open", side_effect=open_descriptor
        ):
            with mock.patch(
                "qualityscaler.job_lock.os.fdopen",
                side_effect=RuntimeError("fdopen failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "fdopen failed"):
                    with JobLock(self.cache):
                        pass
        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError) as caught:
            os.fstat(descriptors[0])
        self.assertEqual(caught.exception.errno, errno.EBADF)

    def test_unlock_error_still_closes_handle_and_releases_ownership(self):
        with mock.patch(
            "qualityscaler.job_lock._unlock_file",
            side_effect=OSError(errno.EIO, "unlock failed"),
        ):
            with self.assertRaises(OSError):
                with JobLock(self.cache):
                    pass
        with JobLock(self.cache):
            pass


if __name__ == "__main__":
    unittest.main()
