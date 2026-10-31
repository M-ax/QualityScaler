import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qualityscaler.self_test import run_self_test


class PackagedSelfTestTests(unittest.TestCase):
    def test_success_writes_report_and_log_without_leaving_test_media(self):
        with TemporaryDirectory() as directory:
            report_path = Path(directory) / "packaged.json"

            def check(root, report):
                (root / "temporary.mp4").write_bytes(b"video")
                report["frames"] = 8
                print("checked")

            with patch("qualityscaler.self_test._check_bundle", side_effect=check):
                self.assertEqual(run_self_test(report_path), 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["frames"], 8)
            self.assertEqual(report_path.with_suffix(".log").read_text(), "checked\n")
            self.assertEqual(set(Path(directory).iterdir()),
                             {report_path, report_path.with_suffix(".log")})

    def test_failure_returns_nonzero_and_records_diagnostics(self):
        with TemporaryDirectory() as directory:
            report_path = Path(directory) / "packaged.json"
            with patch("qualityscaler.self_test._check_bundle",
                       side_effect=RuntimeError("Missing packaged model")):
                self.assertEqual(run_self_test(report_path), 1)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("Missing packaged model", report["error"])
            self.assertIn("Missing packaged model",
                          report_path.with_suffix(".log").read_text())


if __name__ == "__main__":
    unittest.main()
