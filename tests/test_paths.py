import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from qualityscaler.paths import (
    OUTPUT_PATH_CODED,
    find_by_relative_path,
    prepare_output_image_filename,
    prepare_output_video_directory_name,
    prepare_output_video_filename,
)


class ResourcePathTests(unittest.TestCase):
    def test_resources_resolve_against_repository_not_package(self):
        root = Path(__file__).resolve().parents[1]
        with patch.dict(sys.__dict__):
            sys.__dict__.pop("_MEIPASS", None)
            self.assertEqual(
                find_by_relative_path(os.path.join("AI-onnx", "model.onnx")),
                str(root / "AI-onnx" / "model.onnx"),
            )

    def test_resources_resolve_against_pyinstaller_bundle(self):
        bundle = os.path.abspath("bundled resources")
        with patch.object(sys, "_MEIPASS", bundle, create=True):
            self.assertEqual(
                find_by_relative_path(os.path.join("Assets", "exiftool.exe")),
                os.path.join(bundle, "Assets", "exiftool.exe"),
            )

    def test_absolute_resource_path_is_unchanged(self):
        absolute = os.path.abspath(os.path.join("資料", "preference.json"))
        self.assertEqual(find_by_relative_path(absolute), absolute)


class OutputPathTests(unittest.TestCase):
    def test_same_directory_image_and_video_names(self):
        for helper, source_ext, output_ext in (
            (prepare_output_image_filename, ".jpeg", ".png"),
            (prepare_output_video_filename, ".MP4", ".mkv"),
        ):
            with self.subTest(helper=helper.__name__):
                source = os.path.join("input", "holiday.final" + source_ext)
                self.assertEqual(
                    helper(source, OUTPUT_PATH_CODED, "BSRGANx4", 1, 1, output_ext, 0),
                    os.path.join("input", "holiday.final_upscaled" + output_ext),
                )

    def test_custom_output_directory_and_unicode(self):
        source = os.path.join("入力", "猫 🐈.original.mp4")
        destination = os.path.join("出力", "完成 images")
        for helper in (prepare_output_image_filename, prepare_output_video_filename):
            with self.subTest(helper=helper.__name__):
                self.assertEqual(
                    helper(source, destination, "RealESR_Gx4", 0.5, 2, ".png", 0.3),
                    os.path.join(destination, "猫 🐈.original_upscaled.png"),
                )

    def test_model_factors_and_blending_do_not_change_names(self):
        for helper in (prepare_output_image_filename, prepare_output_video_filename):
            for model in ("BSRGANx2", "RealESR_Gx4", "IRCNN_Mx1"):
                for input_factor, output_factor in ((1, 1), (0.1239, 2.999), (2, 0.5)):
                    for blending in (0, 0.3, 0.5, 0.7, 0.4):
                        with self.subTest(
                            helper=helper.__name__,
                            model=model,
                            factors=(input_factor, output_factor),
                            blending=blending,
                        ):
                            self.assertEqual(
                                helper(
                                    "clip.jpg", OUTPUT_PATH_CODED, model,
                                    input_factor, output_factor, ".tiff", blending,
                                ),
                                "clip_upscaled.tiff",
                            )

    def test_cache_directory_is_stable_and_includes_source_extension(self):
        source = os.path.join("入力", "映像.final.mp4")
        for model, input_factor, output_factor, blending in (
            ("BSRGANx4", 1, 1, 0),
            ("RealESR_Gx4", 0.1239, 2.999, 0.7),
        ):
            with self.subTest(model=model):
                self.assertEqual(
                    prepare_output_video_directory_name(
                        source, OUTPUT_PATH_CODED, model,
                        input_factor, output_factor, blending,
                    ),
                    os.path.join("入力", "映像.final.mp4.qualityscaler"),
                )

    def test_custom_cache_directory_and_distinct_source_extensions(self):
        destination = os.path.join("output", "cache")
        names = [
            prepare_output_video_directory_name(
                os.path.join("input", source), destination, "BSRGANx2", 1, 1, 0.5
            )
            for source in ("clip.mp4", "clip.mkv")
        ]
        self.assertEqual(
            names,
            [
                os.path.join(destination, "clip.mp4.qualityscaler"),
                os.path.join(destination, "clip.mkv.qualityscaler"),
            ],
        )

    def test_relative_source_without_parent_directory(self):
        self.assertEqual(
            prepare_output_video_directory_name(
                "clip.mp4", OUTPUT_PATH_CODED, "BSRGANx2", 1, 1, 0
            ),
            "clip.mp4.qualityscaler",
        )


if __name__ == "__main__":
    unittest.main()
