"""Build-time splash artwork uses JPEG compression, not reduced-resolution pixels."""

from base64 import b64decode
from io import BytesIO
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from PIL import Image, ImageChops, ImageStat

from qualityscaler.splash_art import JPEG_STAGES, jpeg_artifact_script


class SplashArtworkTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image_file = Path(self.temp.name) / "logo.png"
        self.source = Image.frombytes("RGB", (96, 64), bytes(
            component
            for y in range(64)
            for x in range(96)
            for component in ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256)
        ))
        self.source.save(self.image_file)

    def frames(self, size):
        script = jpeg_artifact_script(self.image_file, size)
        self.assertTrue(script.isascii())
        data = re.findall(r"image create photo -data \{([A-Za-z0-9+/=]+)\}", script)
        self.assertEqual(len(data), len(JPEG_STAGES))
        images = []
        for encoded in data:
            with Image.open(BytesIO(b64decode(encoded))) as image:
                self.assertEqual(image.format, "PNG")
                images.append(image.copy())
        return images

    def test_every_stage_keeps_native_resolution_and_matches_real_jpeg_round_trip(self):
        frames = self.frames(self.source.size)
        self.assertTrue(all(frame.size == self.source.size for frame in frames))
        for (_, quality, generations), frame in zip(JPEG_STAGES, frames):
            if generations != 1:
                continue
            with self.subTest(quality=quality):
                jpeg = BytesIO()
                self.source.save(jpeg, format="JPEG", quality=quality, subsampling=2)
                jpeg.seek(0)
                with Image.open(jpeg) as expected:
                    self.assertEqual(frame.tobytes(), expected.tobytes())
        self.assertNotEqual(frames[0].tobytes(), self.source.tobytes())

    def test_initial_generation_loss_is_worse_than_a_single_quality_one_jpeg(self):
        frames = self.frames(self.source.size)
        errors = [
            sum(ImageStat.Stat(ImageChops.difference(frame, self.source)).mean)
            for frame in frames[:3]
        ]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        thresholds = [threshold for threshold, _, _ in JPEG_STAGES]
        self.assertEqual(thresholds[0], 0)
        self.assertEqual(thresholds, sorted(thresholds))
        self.assertGreaterEqual(thresholds[1], 0.75)
        self.assertLess(thresholds[-1], 1)

    def test_higher_quality_reduces_artifacts_without_upscaling_small_images(self):
        frames = self.frames((480, 480))
        self.assertTrue(all(frame.size == self.source.size for frame in frames))
        errors = [
            sum(ImageStat.Stat(ImageChops.difference(frame, self.source)).mean)
            for frame in frames
        ]
        self.assertLess(errors[-1], errors[0])

    def test_only_initial_fit_changes_resolution_and_preserves_aspect_ratio(self):
        frames = self.frames((48, 48))
        self.assertTrue(all(frame.size == (48, 32) for frame in frames))


if __name__ == "__main__":
    unittest.main()
