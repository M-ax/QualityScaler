from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qualityscaler.__main__ import main
from qualityscaler.encoders import EncoderOptions
from qualityscaler.frame_store import FrameStore
from qualityscaler.video import VideoSettings


class SavedJobCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / "frames.sqlite3"
        self.settings = VideoSettings("SavedModelx2", blending_factor=0.5, workers=4)
        with FrameStore(self.cache) as store:
            store.set_metadata("source", {"path": "saved input.mp4"})
            store.set_metadata("settings", asdict(self.settings))
            store.set_metadata("output_path", "saved output.mp4")
            store.set_metadata("stage", "paused")
            store.put_frame("source", 0, b"jpeg bytes")

    def test_inspect_reads_settings_and_progress_as_json(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["inspect", str(self.cache)]), 0)
        saved = json.loads(output.getvalue())
        self.assertEqual(saved["settings"], asdict(self.settings))
        self.assertEqual(saved["source_frames"], 1)
        self.assertEqual(saved["upscaled_frames"], 0)
        self.assertEqual(saved["stage"], "paused")

    def test_resume_missing_cache_does_not_create_a_database(self):
        missing = self.cache.with_name("missing.sqlite3")
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            main(["resume", str(missing)])
        self.assertFalse(missing.exists())

    @patch("qualityscaler.__main__.VideoUpscaler")
    def test_resume_restores_saved_settings_and_optional_codec_override(self, pipeline):
        self.assertEqual(main(["resume", str(self.cache), "--codec", "x265"]), 0)
        arguments = pipeline.return_value.run.call_args.args
        self.assertEqual(arguments[:3],
                         ("saved input.mp4", "saved output.mp4", self.cache))
        self.assertEqual(arguments[3].model, "SavedModelx2")
        self.assertEqual(arguments[3].blending_factor, 0.5)
        self.assertEqual(arguments[3].workers, 4)
        self.assertEqual(arguments[3].codec, "x265")

    @patch("qualityscaler.__main__.VideoUpscaler")
    def test_resume_restores_nested_encoder_options_and_normalizes_codec_override(self, pipeline):
        options = EncoderOptions(preset="p7", bitrate_kbps=9000)
        with FrameStore(self.cache) as store:
            settings = asdict(self.settings)
            settings.update(codec="h264_nvenc", encoder_options=asdict(options))
            store.set_metadata("settings", settings)
        self.assertEqual(main(["resume", str(self.cache)]), 0)
        self.assertEqual(pipeline.return_value.run.call_args.args[3].encoder_options, options)
        self.assertEqual(main(["resume", str(self.cache), "--codec", "x264"]), 0)
        self.assertEqual(pipeline.return_value.run.call_args.args[3].encoder_options,
                         EncoderOptions(bitrate_kbps=9000))

    @patch("qualityscaler.__main__.VideoUpscaler")
    def test_resume_legacy_settings_uses_original_bitrate_default(self, pipeline):
        with FrameStore(self.cache) as store:
            settings = asdict(self.settings)
            settings.pop("encoder_options")
            store.set_metadata("settings", settings)
        self.assertEqual(main(["resume", str(self.cache)]), 0)
        self.assertEqual(pipeline.return_value.run.call_args.args[3].encoder_options, EncoderOptions())


if __name__ == "__main__":
    unittest.main()
