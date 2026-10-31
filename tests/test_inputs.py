import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qualityscaler.formats import INPUT_EXTENSIONS
from qualityscaler.inputs import merge_input_files, supported_input_files


class InputFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def file(self, name):
        path = self.root / name
        path.write_bytes(b"fixture")
        return str(path)

    def test_supported_extensions_are_case_insensitive(self):
        paths = [self.file("input" + suffix.upper()) for suffix in sorted(INPUT_EXTENSIONS)]
        self.assertEqual(supported_input_files(paths), paths)

    def test_only_final_extension_counts(self):
        valid = self.file("photo.PNG")
        invalid = self.file("movie.mp4.exe")
        directory = self.root / "videos.mp4"
        directory.mkdir()
        hidden_text = directory / "notes.txt"
        hidden_text.write_text("not media")
        self.assertEqual(supported_input_files([invalid, str(hidden_text), valid]), [valid])

    def test_directories_missing_files_and_invalid_values_are_ignored(self):
        directory = self.root / "folder.png"
        directory.mkdir()
        self.assertEqual(supported_input_files([
            str(directory), str(self.root / "missing.mp4"), None, 42, "",
        ]), [])

    def test_unicode_spaces_braces_and_apostrophes_are_preserved(self):
        path = self.file("résumé's image {one} with spaces.png")
        self.assertEqual(supported_input_files([path]), [path])

    def test_canonical_paths_are_deduplicated_in_original_order(self):
        first = self.file("first.jpg")
        second = self.file("second.mp4")
        nested = self.root / "nested"
        nested.mkdir()
        alias = str(nested / ".." / Path(first).name)
        paths = [first, alias, second, first]
        if os.name == "nt":
            paths.append(first.upper())
        self.assertEqual(supported_input_files(paths), [first, second])

    def test_merge_appends_and_deduplicates_instead_of_replacing(self):
        first = self.file("first.jpg")
        second = self.file("second.mp4")
        third = self.file("third.TIFF")
        self.assertEqual(merge_input_files([first, second], [second, third, first]),
                         [first, second, third])


if __name__ == "__main__":
    unittest.main()
