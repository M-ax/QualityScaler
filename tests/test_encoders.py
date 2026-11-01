from dataclasses import FrozenInstanceError
import subprocess
import sys
import unittest

from qualityscaler.encoders import (
    EncoderOptions, encoder_capabilities, normalize_encoder_options,
)


SOFTWARE_CODECS = ("x264", "x265", "libx264", "libx265")
HARDWARE_CODECS = (
    "h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf", "h264_qsv", "hevc_qsv",
)


class EncoderCapabilitiesTests(unittest.TestCase):
    def test_module_imports_without_external_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", "-c", "import qualityscaler.encoders"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_software_capabilities_and_aliases(self):
        expected = (
            "Default", "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow", "placebo",
        )
        for codec in SOFTWARE_CODECS:
            with self.subTest(codec=codec):
                capabilities = encoder_capabilities(codec)
                self.assertEqual(capabilities.presets, expected)
                self.assertEqual(capabilities.rate_controls, ("Bitrate", "CRF"))
                self.assertEqual(capabilities.preset_flag, "-preset")
        self.assertEqual(encoder_capabilities("x264"), encoder_capabilities("libx264"))
        self.assertEqual(encoder_capabilities("x265"), encoder_capabilities("libx265"))

    def test_hardware_presets_are_encoder_specific_and_never_offer_crf(self):
        families = {
            "nvenc": (("Default", "p1", "p2", "p3", "p4", "p5", "p6", "p7"), "-preset"),
            "amf": (("Default", "speed", "balanced", "quality"), "-quality"),
            "qsv": (("Default", "veryfast", "faster", "fast", "medium", "slow",
                     "slower", "veryslow"), "-preset"),
        }
        for codec in HARDWARE_CODECS:
            with self.subTest(codec=codec):
                expected_presets, flag = families[codec.rsplit("_", 1)[1]]
                capabilities = encoder_capabilities(codec)
                self.assertEqual(capabilities.presets, expected_presets)
                self.assertEqual(capabilities.rate_controls, ("Bitrate",))
                self.assertEqual(capabilities.preset_flag, flag)

    def test_capabilities_are_immutable(self):
        capabilities = encoder_capabilities("x264")
        with self.assertRaises(FrozenInstanceError):
            capabilities.presets = ("something else",)

    def test_unknown_codec_only_exposes_legacy_bitrate_default(self):
        capabilities = encoder_capabilities("not_a_video_codec")
        self.assertEqual(capabilities.presets, ("Default",))
        self.assertEqual(capabilities.rate_controls, ("Bitrate",))
        self.assertIsNone(capabilities.preset_flag)
        self.assertEqual(EncoderOptions().arguments("not_a_video_codec"), ["-b:v", "50000k"])


class EncoderOptionsTests(unittest.TestCase):
    def test_defaults_preserve_legacy_bitrate_without_forcing_a_preset(self):
        options = EncoderOptions()
        self.assertEqual(
            (options.preset, options.rate_control, options.crf, options.bitrate_kbps),
            ("Default", "Bitrate", 23.0, 50000),
        )
        for codec in SOFTWARE_CODECS + HARDWARE_CODECS:
            with self.subTest(codec=codec):
                self.assertEqual(options.arguments(codec), ["-b:v", "50000k"])
        with self.assertRaises(FrozenInstanceError):
            options.crf = 20

    def test_every_supported_explicit_preset_uses_its_encoder_flag(self):
        for codec in SOFTWARE_CODECS + HARDWARE_CODECS:
            capabilities = encoder_capabilities(codec)
            for preset in capabilities.presets[1:]:
                with self.subTest(codec=codec, preset=preset):
                    self.assertEqual(
                        EncoderOptions(preset=preset, bitrate_kbps=1234).arguments(codec),
                        [capabilities.preset_flag, preset, "-b:v", "1234k"],
                    )

    def test_crf_has_no_bitrate_flag_and_preserves_fractional_value(self):
        for codec in SOFTWARE_CODECS:
            with self.subTest(codec=codec):
                self.assertEqual(
                    EncoderOptions(preset="slow", rate_control="CRF", crf=18.25).arguments(codec),
                    ["-preset", "slow", "-crf", "18.25"],
                )

    def test_unknown_or_cross_encoder_presets_are_rejected(self):
        for codec, preset in (
            ("x264", "p7"), ("x265", "quality"), ("h264_nvenc", "slow"),
            ("hevc_amf", "p1"), ("h264_qsv", "ultrafast"),
            ("x264", "default"), ("x264", ""), ("x264", None),
        ):
            with self.subTest(codec=codec, preset=preset):
                with self.assertRaisesRegex(ValueError, "Unsupported preset"):
                    EncoderOptions(preset=preset).arguments(codec)

    def test_hardware_crf_and_unknown_modes_are_rejected(self):
        for codec in HARDWARE_CODECS:
            with self.subTest(codec=codec):
                with self.assertRaisesRegex(ValueError, "Unsupported rate control"):
                    EncoderOptions(rate_control="CRF").validate(codec)
        for mode in ("", "CQ", "QP", "crf", "CBR", None):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "Unsupported rate control"):
                    EncoderOptions(rate_control=mode).validate("x264")

    def test_crf_validation_accepts_inclusive_bounds_and_finite_fractions(self):
        for value in (0, 0.0, 0.125, 23, 23.0, 50.99, 51, 51.0):
            for codec in SOFTWARE_CODECS:
                with self.subTest(codec=codec, value=value):
                    EncoderOptions(rate_control="CRF", crf=value).validate(codec)

    def test_crf_validation_rejects_nonfinite_out_of_range_and_nonnumeric_values(self):
        for value in (
            -0.001, 51.001, float("nan"), float("inf"), -float("inf"),
            10 ** 1000, "23", None, True, False,
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "CRF.*finite.*0 to 51"):
                    EncoderOptions(rate_control="CRF", crf=value).arguments("libx265")

    def test_bitrate_must_be_positive_whole_kbps(self):
        for value in (1, 50000, 250000):
            with self.subTest(value=value):
                EncoderOptions(bitrate_kbps=value).validate("h264_nvenc")
        for value in (
            0, -1, 0.5, 50000.0, float("nan"), float("inf"),
            "50000", None, True, False,
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Bitrate.*positive whole.*kb/s"):
                    EncoderOptions(bitrate_kbps=value).arguments("x264")

    def test_only_the_active_numeric_value_is_validated(self):
        for inactive in (None, "unparsed", -1, float("nan"), float("inf"), True):
            with self.subTest(mode="Bitrate", inactive=inactive):
                for codec in SOFTWARE_CODECS + HARDWARE_CODECS:
                    self.assertEqual(
                        EncoderOptions(crf=inactive).arguments(codec),
                        ["-b:v", "50000k"],
                    )
            with self.subTest(mode="CRF", inactive=inactive):
                self.assertEqual(
                    EncoderOptions(rate_control="CRF", crf=20, bitrate_kbps=inactive)
                    .arguments("x264"),
                    ["-crf", "20"],
                )


class NormalizeEncoderOptionsTests(unittest.TestCase):
    def test_supported_choices_and_all_numeric_values_are_retained(self):
        options = EncoderOptions("slow", "CRF", 17.5, 8500)
        normalized = normalize_encoder_options("libx264", options)
        self.assertEqual(normalized, options)
        self.assertEqual(options, EncoderOptions("slow", "CRF", 17.5, 8500))

    def test_software_to_hardware_resets_only_incompatible_choices(self):
        options = EncoderOptions("slow", "CRF", 17.5, 8500)
        self.assertEqual(
            normalize_encoder_options("h264_nvenc", options),
            EncoderOptions("Default", "Bitrate", 17.5, 8500),
        )
        self.assertEqual(
            normalize_encoder_options("h264_qsv", options),
            EncoderOptions("slow", "Bitrate", 17.5, 8500),
        )

    def test_gpu_fallback_preserves_bitrate_and_only_compatible_presets(self):
        for codec, preset, expected in (
            ("h264_nvenc", "p7", "Default"),
            ("hevc_amf", "quality", "Default"),
            ("h264_qsv", "veryslow", "veryslow"),
        ):
            with self.subTest(codec=codec):
                options = EncoderOptions(preset=preset, bitrate_kbps=7200)
                options.validate(codec)
                fallback = normalize_encoder_options("libx264", options)
                self.assertEqual(fallback, EncoderOptions(preset=expected, bitrate_kbps=7200))
                self.assertNotIn("-quality", fallback.arguments("libx264"))
                self.assertEqual(options.preset, preset)

    def test_normalization_does_not_repair_or_validate_numeric_values(self):
        options = EncoderOptions("p7", "CRF", float("inf"), -1)
        normalized = normalize_encoder_options("h264_amf", options)
        self.assertEqual(normalized, EncoderOptions("Default", "Bitrate", float("inf"), -1))
        with self.assertRaisesRegex(ValueError, "Bitrate"):
            normalized.validate("h264_amf")
        normalized = normalize_encoder_options("x264", options)
        self.assertEqual(normalized.rate_control, "CRF")
        self.assertEqual(normalized.crf, float("inf"))
        self.assertEqual(normalized.bitrate_kbps, -1)
        with self.assertRaisesRegex(ValueError, "CRF"):
            normalized.validate("x264")

    def test_unknown_choices_normalize_without_touching_numeric_preferences(self):
        self.assertEqual(
            normalize_encoder_options("x264", EncoderOptions("invalid", "invalid", 19, 2500)),
            EncoderOptions("Default", "Bitrate", 19, 2500),
        )


if __name__ == "__main__":
    unittest.main()
