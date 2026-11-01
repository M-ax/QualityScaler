"""Dimension policy without image buffers, GUI imports, or inference sessions."""

import unittest

from qualityscaler.geometry import VideoGeometry, scaled_size


class GeometryTests(unittest.TestCase):
    def test_shared_input_rule_truncates_then_rounds_odd_values_up(self):
        for width, height, factor, expected in (
            (15, 11, 0.5, (8, 6)),
            (7, 5, 1, (8, 6)),
            (1024, 1024, 0.5, (512, 512)),
            (35, 27, 2, (70, 54)),
        ):
            with self.subTest(size=(width, height), factor=factor):
                self.assertEqual(scaled_size(width, height, factor), expected)

    def test_output_geometry_is_not_recomputed_from_already_scaled_frames(self):
        geometry = VideoGeometry.calculate(100, 100, 0.53, 2, 1)
        self.assertEqual(geometry.source_size, (100, 100))
        self.assertEqual(geometry.inference_size, (54, 54))
        self.assertEqual(geometry.output_size, (104, 104))
        self.assertNotEqual(geometry.output_size, (108, 108))

    def test_aspect_derived_final_height_matches_existing_video_output(self):
        geometry = VideoGeometry.calculate(15, 11, 0.5, 2, 1.25)
        self.assertEqual(geometry.inference_size, (8, 6))
        self.assertEqual(geometry.output_size, (20, 16))

    def test_tiny_video_input_is_clamped_to_two_pixels_per_axis(self):
        geometry = VideoGeometry.calculate(3, 5, 0.001, 2, 1)
        self.assertEqual(geometry.inference_size, (2, 2))
        self.assertEqual(geometry.output_size, (4, 8))

    def test_missing_source_dimensions_are_rejected(self):
        for width, height in ((0, 10), (10, 0), (-1, 10)):
            with self.subTest(size=(width, height)), self.assertRaises(ValueError):
                VideoGeometry.calculate(width, height, 1, 2, 1)


if __name__ == "__main__":
    unittest.main()
