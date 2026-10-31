"""Exercise the spec without PyInstaller, GUI imports, or real model/tool files."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch, sentinel


ROOT = Path(__file__).resolve().parents[1]
SPEC_CODE = compile(
    (ROOT / "QualityScaler.spec").read_text(encoding="utf-8"),
    str(ROOT / "QualityScaler.spec"),
    "exec",
)
STANDARD_MODELS = (
    "LVAx2",
    "RealESR_Gx4",
    "RealESR_Ax4",
    "BSRGANx2",
    "BSRGANx4",
    "RealESRGANx4",
    "MSharpx4",
    "IRCNN_Mx1",
    "IRCNN_Lx1",
)
REQUIRED_FILES = (
    "QualityScaler.py",
    "Assets/logo.ico",
    "Assets/logo.png",
    "Assets/ffmpeg.exe",
    "LICENSE",
    "LICENSE.txt",
) + tuple(f"AI-onnx/{name}_fp16.onnx" for name in STANDARD_MODELS)


class BuildSpecTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="qualityscaler build ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name in REQUIRED_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

        self.ctk_data = [
            ("venv/customtkinter/assets/themes/blue.json", "customtkinter/assets/themes"),
            ("venv/customtkinter/assets/fonts/font.otf", "customtkinter/assets/fonts"),
            ("venv/customtkinter/assets/icons/image.png", "customtkinter/assets/icons"),
        ]
        self.tkdnd_package = self.root / "venv/tkinterdnd2"
        self.tkdnd_hook_dir = str(self.tkdnd_package / "__pyinstaller")
        self.tkdnd_data = [
            (str(self.tkdnd_package / "tkdnd/win-x64" / name), "tkinterdnd2/tkdnd/win-x64")
            for name in (
                "pkgIndex.tcl", "tkdnd.tcl", "tkdnd_windows.tcl", "libtkdnd2.10.3.dll",
            )
        ]
        self.ort_binaries = [
            (str(self.root / "venv/onnxruntime/capi" / name), "onnxruntime/capi")
            for name in (
                "onnxruntime_providers_shared.dll", "DirectML.dll", "onnxruntime.dll",
            )
        ]
        self.hooks = ModuleType("PyInstaller.utils.hooks")
        self.hooks.collect_data_files = Mock(return_value=self.ctk_data)
        self.hooks.collect_dynamic_libs = Mock(return_value=self.ort_binaries)
        self.hooks.get_package_paths = Mock(return_value=(
            str(self.root / "venv"), str(self.tkdnd_package),
        ))
        modules = {
            "PyInstaller": ModuleType("PyInstaller"),
            "PyInstaller.utils": ModuleType("PyInstaller.utils"),
            "PyInstaller.utils.hooks": self.hooks,
        }
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

        def analyze(*args, **kwargs):
            # The supported tkinterdnd2 hook runs inside Analysis, not the spec.
            # Its selected native DLL is a data entry alongside the Tcl scripts.
            hook_data = (
                self.tkdnd_data if self.tkdnd_hook_dir in kwargs["hookspath"] else []
            )
            return SimpleNamespace(
                pure=sentinel.pure,
                scripts=sentinel.scripts,
                binaries=kwargs["binaries"],
                datas=kwargs["datas"] + hook_data,
            )

        self.analysis = Mock(side_effect=analyze)
        self.pyz = Mock(return_value=sentinel.pyz)
        self.exe = Mock()

    def execute_spec(self):
        namespace = {
            "SPECPATH": str(self.root),
            "Analysis": self.analysis,
            "PYZ": self.pyz,
            "EXE": self.exe,
        }
        exec(SPEC_CODE, namespace)
        return namespace

    def test_standard_model_catalog_contains_all_nine_current_models(self):
        namespace = self.execute_spec()
        self.assertEqual(namespace["STANDARD_MODELS"], STANDARD_MODELS)
        bundled_models = {
            Path(source).name
            for source, destination in self.analysis.call_args.kwargs["datas"]
            if destination == "AI-onnx"
        }
        self.assertEqual(
            bundled_models, {f"{name}_fp16.onnx" for name in STANDARD_MODELS}
        )

    def test_resources_preserve_bundle_paths_and_include_all_fp16_models(self):
        extra_files = [
            "Assets/another_icon.ico", "Assets/upscale_icon.png",
            "Assets/exiftool.exe", "Assets/not_requested.exe", "Assets/readme.txt",
            "AI-onnx/optional_model_fp16.onnx", "AI-onnx/ignored_fp32.onnx",
            "AI-onnx/download.txt",
        ]
        for name in extra_files:
            (self.root / name).write_bytes(b"fixture")
        (self.root / "Assets/not_a_file.png").mkdir()
        (self.root / "AI-onnx/not_a_file_fp16.onnx").mkdir()

        # Analysis must inspect the entry point, never execute/import it in the spec.
        (self.root / "QualityScaler.py").write_text(
            "raise AssertionError('launcher must not be imported by spec')",
            encoding="utf-8",
        )
        self.execute_spec()
        self.hooks.collect_data_files.assert_called_once_with("customtkinter")
        self.hooks.collect_dynamic_libs.assert_called_once_with("onnxruntime")
        args, options = self.analysis.call_args
        self.assertEqual(args, ([str(self.root / "QualityScaler.py")],))
        self.assertEqual(options["pathex"], [str(self.root)])
        self.assertEqual(options["hiddenimports"], [])
        self.assertEqual(options["runtime_hooks"], [])
        self.assertEqual(options["binaries"], sorted(self.ort_binaries))

        expected_resources = {
            (str(self.root / name), destination)
            for name, destination in (
                ("Assets/logo.ico", "Assets"),
                ("Assets/logo.png", "Assets"),
                ("Assets/another_icon.ico", "Assets"),
                ("Assets/upscale_icon.png", "Assets"),
                ("Assets/ffmpeg.exe", "Assets"),
                ("LICENSE", "."),
                ("LICENSE.txt", "."),
            )
        }
        expected_resources.update(
            (str(path), "AI-onnx")
            for path in (self.root / "AI-onnx").glob("*_fp16.onnx")
            if path.is_file()
        )
        self.assertEqual(
            set(options["datas"]), expected_resources | set(self.ctk_data)
        )
        self.assertEqual(len(options["datas"]), len(set(options["datas"])))

    def test_legacy_anime_model_cannot_replace_current_ax4_model(self):
        (self.root / "AI-onnx/RealESR_Ax4_fp16.onnx").unlink()
        (self.root / "AI-onnx/RealESR_Animex4_fp16.onnx").write_bytes(b"legacy")
        with self.assertRaises(SystemExit) as caught:
            self.execute_spec()
        self.assertIn(str(Path("AI-onnx/RealESR_Ax4_fp16.onnx")), str(caught.exception))
        self.hooks.collect_data_files.assert_not_called()
        self.hooks.collect_dynamic_libs.assert_not_called()
        self.hooks.get_package_paths.assert_not_called()
        self.analysis.assert_not_called()

    def test_supported_tkdnd_hook_resources_reach_onefile_with_package_paths(self):
        # Hook discovery must not import the GUI package or create a Tk root.
        with patch.dict(sys.modules, {"tkinterdnd2": None}):
            self.execute_spec()
        self.hooks.get_package_paths.assert_called_once_with("tkinterdnd2")
        self.assertEqual(
            self.analysis.call_args.kwargs["hookspath"], [self.tkdnd_hook_dir]
        )
        bundled_data = self.exe.call_args.args[3]
        for resource in self.tkdnd_data:
            with self.subTest(resource=resource[0]):
                self.assertIn(resource, bundled_data)
                self.assertEqual(bundled_data.count(resource), 1)

    def test_build_is_single_file_windowed_with_icon_and_no_upx(self):
        self.execute_spec()
        self.pyz.assert_called_once_with(sentinel.pure)
        args, options = self.exe.call_args
        self.assertEqual(args, (
            sentinel.pyz,
            sentinel.scripts,
            sorted(self.ort_binaries),
            self.analysis.call_args.kwargs["datas"] + self.tkdnd_data,
            [],
        ))
        self.assertEqual(options["name"], "QualityScaler")
        self.assertEqual(options["icon"], str(self.root / "Assets/logo.ico"))
        self.assertIs(options["console"], False)
        self.assertIs(options["upx"], False)
        self.assertNotIn("exclude_binaries", options)

    def test_each_required_local_file_fails_before_collection_or_analysis(self):
        for name in REQUIRED_FILES:
            with self.subTest(missing=name):
                path = self.root / name
                path.unlink()
                try:
                    with self.assertRaises(SystemExit) as caught:
                        self.execute_spec()
                    self.assertIn(str(Path(name)), str(caught.exception))
                    self.hooks.collect_data_files.assert_not_called()
                    self.hooks.collect_dynamic_libs.assert_not_called()
                    self.hooks.get_package_paths.assert_not_called()
                    self.analysis.assert_not_called()
                finally:
                    path.write_bytes(b"fixture")

    def test_missing_resources_are_reported_together(self):
        for name in ("Assets/ffmpeg.exe", "AI-onnx/RealESR_Gx4_fp16.onnx"):
            (self.root / name).unlink()
        with self.assertRaises(SystemExit) as caught:
            self.execute_spec()
        self.assertIn(str(Path("Assets/ffmpeg.exe")), str(caught.exception))
        self.assertIn(str(Path("AI-onnx/RealESR_Gx4_fp16.onnx")), str(caught.exception))

    def test_each_required_directml_dll_is_checked_before_analysis(self):
        for missing in self.ort_binaries:
            with self.subTest(missing=missing[0]):
                self.hooks.collect_dynamic_libs.return_value = [
                    binary for binary in self.ort_binaries if binary != missing
                ]
                with self.assertRaises(SystemExit) as caught:
                    self.execute_spec()
                self.assertIn(Path(missing[0]).name.lower(), str(caught.exception))
                self.assertIn("requirements.txt", str(caught.exception))
                self.hooks.get_package_paths.assert_not_called()
                self.analysis.assert_not_called()


if __name__ == "__main__":
    unittest.main()
