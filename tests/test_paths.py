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
                find_by_relative_path(os.path.join("Assets", "logo.png")),
                os.path.join(bundle, "Assets", "logo.png"),
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

    def test_model_factors_and_sharpening_do_not_change_names(self):
        for helper in (prepare_output_image_filename, prepare_output_video_filename):
            for model in ("BSRGANx2", "RealESR_Gx4", "IRCNN_Mx1"):
                for input_factor, output_factor in ((1, 1), (0.1239, 2.999), (2, 0.5)):
                    for sharpening in (0, 0.3, 0.5):
                        with self.subTest(
                            helper=helper.__name__,
                            model=model,
                            factors=(input_factor, output_factor),
                            sharpening=sharpening,
                        ):
                            self.assertEqual(
                                helper(
                                    "clip.jpg", OUTPUT_PATH_CODED, model,
                                    input_factor, output_factor, ".tiff", sharpening,
                                ),
                                "clip_upscaled.tiff",
                            )

    def test_filename_helpers_accept_canonical_sharpening_keyword(self):
        for helper, source_keyword, extension_keyword in (
            (prepare_output_image_filename, "image_path", "selected_image_extension"),
            (prepare_output_video_filename, "video_path", "selected_video_extension"),
        ):
            with self.subTest(helper=helper.__name__):
                self.assertEqual(
                    helper(
                        **{
                            source_keyword: os.path.join("input", "clip.original.jpg"),
                            "selected_output_path": "output",
                            "selected_AI_model": "RealESR_Gx4",
                            "input_resize_factor": 0.5,
                            "output_resize_factor": 2,
                            extension_keyword: ".png",
                            "selected_sharpening_amount": 0.3,
                        }
                    ),
                    os.path.join("output", "clip.original_upscaled.png"),
                )

    def test_cache_directory_is_stable_and_includes_source_extension(self):
        source = os.path.join("入力", "映像.final.mp4")
        for model, input_factor, output_factor, sharpening in (
            ("BSRGANx4", 1, 1, 0),
            ("RealESR_Gx4", 0.1239, 2.999, 0.5),
        ):
            with self.subTest(model=model):
                self.assertEqual(
                    prepare_output_video_directory_name(
                        source, OUTPUT_PATH_CODED, model,
                        input_factor, output_factor, sharpening,
                    ),
                    os.path.join("入力", "映像.final.mp4.qualityscaler"),
                )

    def test_cache_directory_accepts_canonical_sharpening_keyword(self):
        self.assertEqual(
            prepare_output_video_directory_name(
                video_path=os.path.join("input", "clip.mp4"),
                selected_output_path=OUTPUT_PATH_CODED,
                selected_AI_model="RealESR_Gx4",
                input_resize_factor=0.5,
                output_resize_factor=2,
                selected_sharpening_amount=0.5,
            ),
            os.path.join("input", "clip.mp4.qualityscaler"),
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
