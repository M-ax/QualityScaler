import ctypes
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from qualityscaler.gpu import (
    GPU, get_AI_engine_info, get_AI_providers, get_available_AI_providers,
    is_directml_available,
)


class ProviderTests(unittest.TestCase):
    def test_provider_preference_and_engine_info(self):
        for providers, expected in (
            (["CPUExecutionProvider", "DmlExecutionProvider"], "DirectML"),
            (["DirectMLExecutionProvider"], "DirectML"),
            (["CPUExecutionProvider"], "CPU"),
            (["CUDAExecutionProvider", "CPUExecutionProvider"], "CPU"),
            ([], "CPU"),
        ):
            with self.subTest(providers=providers), patch.dict(
                sys.modules, {"onnxruntime": SimpleNamespace(
                    get_available_providers=lambda: providers,
                    get_version_string=lambda: "1.24.4",
                )},
            ):
                self.assertEqual(get_available_AI_providers(), providers)
                self.assertEqual(is_directml_available(), expected == "DirectML")
                self.assertEqual(
                    get_AI_providers(),
                    ["DmlExecutionProvider" if expected == "DirectML" else "CPUExecutionProvider"],
                )
                self.assertEqual(get_AI_engine_info(), f"AI engine 1.24.4 + {expected}")

    def test_unavailable_runtime_falls_back_to_cpu_and_empty_engine_label(self):
        with patch.dict(sys.modules, {"onnxruntime": None}):
            self.assertEqual(get_available_AI_providers(), [])
            self.assertFalse(is_directml_available())
            self.assertEqual(get_AI_providers(), ["CPUExecutionProvider"])
            self.assertEqual(get_AI_engine_info(), "")

    def test_broken_provider_query_and_version_are_optional(self):
        with patch.dict(sys.modules, {"onnxruntime": SimpleNamespace(
            get_available_providers=Mock(side_effect=RuntimeError("no runtime")),
            get_version_string=Mock(side_effect=RuntimeError("no version")),
        )}):
            self.assertEqual(get_available_AI_providers(), [])
            self.assertEqual(get_AI_providers(), ["CPUExecutionProvider"])
            self.assertEqual(get_AI_engine_info(), "")


class RegistryTests(unittest.TestCase):
    def setUp(self):
        registry = patch.object(GPU, "detected", [])
        registry.start()
        self.addCleanup(registry.stop)

    def test_empty_registry_uses_placeholder_and_auto_device(self):
        self.assertEqual(GPU.menu_list(), ["No GPU found"])
        self.assertEqual(GPU.default(), "No GPU found")
        self.assertEqual(GPU.device_id_for("Unknown GPU"), "Auto")
        self.assertEqual(GPU.device_id_for("Auto"), "Auto")
        self.assertIsNone(GPU.find("Unknown GPU"))
        self.assertIsNone(GPU.vram_for(GPU.NO_GPU))
        self.assertIsNone(GPU.codec_for(GPU.NO_GPU))

    def test_named_gpu_map_vram_and_vendor_codec(self):
        GPU.detected = [
            GPU("RTX 5060 Ti", 15.9, 0, GPU.VENDOR_NVIDIA),
            GPU("RX 7900 XTX", 24, 1, GPU.VENDOR_AMD),
            GPU("Iris Xe Graphics", 0, 2, GPU.VENDOR_INTEL),
            GPU("Unknown", 1.9, 3),
        ]
        self.assertEqual(GPU.menu_list(), [gpu.name for gpu in GPU.detected])
        self.assertEqual(GPU.default(), "RTX 5060 Ti")
        for gpu, expected_vram, expected_codec in zip(
            GPU.detected, [15, 24, 1, 1], ["h264_nvenc", "h264_amf", "h264_qsv", None],
        ):
            with self.subTest(gpu=gpu):
                self.assertIs(GPU.find(gpu.name), gpu)
                self.assertEqual(GPU.device_id_for(gpu.name), str(gpu.device_id))
                self.assertEqual(GPU.vram_for(gpu.name), expected_vram)
                self.assertEqual(GPU.codec_for(gpu.name), expected_codec)
        # Match upstream: stale names choose the first adapter, but unknown
        # selections must not inherit that adapter's VRAM or codec.
        self.assertEqual(GPU.device_id_for("stale name"), "0")
        self.assertIsNone(GPU.vram_for("stale name"))
        self.assertIsNone(GPU.codec_for("stale name"))

    def test_detection_replaces_stale_registry(self):
        GPU.detected = [GPU("Stale GPU", 8, 0)]
        detected = [GPU("Iris Xe Graphics", 0, 0, GPU.VENDOR_INTEL)]
        with patch.object(GPU, "_enumerate", return_value=detected) as enumerate_gpus:
            GPU.detect()
        enumerate_gpus.assert_called_once_with()
        self.assertIs(GPU.detected, detected)
        with patch.object(GPU, "_enumerate", return_value=[]):
            GPU.detect()
        self.assertEqual(GPU.menu_list(), [GPU.NO_GPU])

    def test_short_names_remove_branding_but_preserve_readable_integrated_names(self):
        for raw, expected in (
            ("NVIDIA GeForce RTX 5060 Ti", "RTX 5060 Ti"),
            ("AMD Radeon RX 7900 XTX", "RX 7900 XTX"),
            ("Intel(R) Iris(R) Xe Graphics", "Iris Xe Graphics"),
            ("AMD Radeon(TM) Graphics", "Radeon Graphics"),
            ("Advanced Micro Devices Radeon Vega 7", "Vega 7"),
            ("Intel® UHD™ Graphics© 770", "UHD Graphics 770"),
            ("NVIDIA", "NVIDIA"),
            ("", ""),
            ("  Custom  GPU  ", "Custom GPU"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(GPU._shorten_name(raw), expected)


class FakeDXGI:
    """A fake COM vtable; discovery never invokes a DLL or a real adapter."""

    def __init__(self, descriptions, *, factory_result=0, desc_result=0):
        self.descriptions = descriptions
        self.factory_result = factory_result
        self.desc_result = desc_result
        self.factory_release = Mock(return_value=0)
        self.adapter_release = Mock(return_value=0)
        self.factory_table = (ctypes.c_void_p * 13)()
        self.factory_table[12] = 101
        self.factory_table[2] = 102
        self.factory_object = (ctypes.c_void_p * 1)(ctypes.addressof(self.factory_table))
        self.adapter_table = (ctypes.c_void_p * 11)()
        self.adapter_table[10] = 103
        self.adapter_table[2] = 104
        self.adapter_objects = [
            (ctypes.c_void_p * 1)(ctypes.addressof(self.adapter_table))
            for _ in descriptions
        ]
        self.adapter_indices = {
            ctypes.addressof(adapter): index for index, adapter in enumerate(self.adapter_objects)
        }
        self.methods = {
            101: self.enum_adapters,
            102: self.factory_release,
            103: self.get_desc,
            104: self.adapter_release,
        }
        self.windll = SimpleNamespace(dxgi=SimpleNamespace(CreateDXGIFactory1=self.create_factory))

    def create_factory(self, iid, factory_out):
        if self.factory_result == 0:
            factory_out._obj.value = ctypes.addressof(self.factory_object)
        return self.factory_result

    def enum_adapters(self, factory, index, adapter_out):
        if index == len(self.adapter_objects):
            return 1  # DXGI_ERROR_NOT_FOUND (only its nonzero status matters).
        adapter_out._obj.value = ctypes.addressof(self.adapter_objects[index])
        return 0

    def get_desc(self, adapter, desc_out):
        if self.desc_result != 0:
            return self.desc_result
        data = self.descriptions[self.adapter_indices[adapter.value]]
        desc = desc_out._obj
        desc.Description = data["name"]
        desc.DedicatedVideoMemory = int(data.get("vram_gb", 0) * 1024 ** 3)
        desc.VendorId = data.get("vendor_id", 0)
        desc.Flags = data.get("flags", 0)
        return 0

    def prototype(self, restype, *argtypes):
        return lambda address: self.methods[address]

    def enumerate(self):
        with patch("qualityscaler.gpu.sys.platform", "win32"), \
                patch.object(ctypes, "windll", self.windll, create=True), \
                patch.object(ctypes, "WINFUNCTYPE", self.prototype, create=True):
            return GPU._enumerate()


class DXGIEnumerationTests(unittest.TestCase):
    def test_hardware_indices_skip_software_and_disambiguate_identical_names(self):
        dxgi = FakeDXGI([
            {"name": "Microsoft Basic Render Driver", "flags": 2},
            {"name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 15.95, "vendor_id": 0x10DE},
            {"name": "Microsoft WARP", "flags": 2},
            {"name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 8.25, "vendor_id": 0x10DE},
            {"name": "Intel(R) Iris(R) Xe Graphics", "vendor_id": 0x8086},
            {"name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 16, "vendor_id": 0x10DE},
        ])
        gpus = dxgi.enumerate()
        self.assertEqual([gpu.name for gpu in gpus], [
            "RTX 5060 Ti", "RTX 5060 Ti (2)", "Iris Xe Graphics", "RTX 5060 Ti (3)",
        ])
        self.assertEqual([gpu.device_id for gpu in gpus], [0, 1, 2, 3])
        self.assertEqual([gpu.vram_gb for gpu in gpus], [15.9, 8.2, 0, 16])
        self.assertEqual([gpu.hardware_codec for gpu in gpus], [
            "h264_nvenc", "h264_nvenc", "h264_qsv", "h264_nvenc",
        ])
        self.assertEqual(dxgi.adapter_release.call_count, 6)
        dxgi.factory_release.assert_called_once()
        with patch.object(GPU, "detected", gpus):
            self.assertEqual(GPU.device_id_for("RTX 5060 Ti (2)"), "1")
            self.assertEqual(GPU.vram_for("RTX 5060 Ti (2)"), 8)

    def test_non_windows_does_not_try_dxgi(self):
        with patch("qualityscaler.gpu.sys.platform", "linux"), \
                patch.object(ctypes, "windll", None, create=True):
            self.assertEqual(GPU._enumerate(), [])

    def test_factory_failure_is_optional(self):
        dxgi = FakeDXGI([], factory_result=-1)
        self.assertEqual(dxgi.enumerate(), [])
        dxgi.factory_release.assert_not_called()

    def test_description_failure_releases_adapter_and_factory(self):
        dxgi = FakeDXGI([{"name": "Unusable"}], desc_result=-1)
        with self.assertLogs("qualityscaler.gpu", level="WARNING"):
            self.assertEqual(dxgi.enumerate(), [])
        dxgi.adapter_release.assert_called_once()
        dxgi.factory_release.assert_called_once()

    def test_missing_dxgi_is_optional(self):
        with patch("qualityscaler.gpu.sys.platform", "win32"), \
                patch.object(ctypes, "windll", SimpleNamespace(), create=True), \
                self.assertLogs("qualityscaler.gpu", level="WARNING"):
            self.assertEqual(GPU._enumerate(), [])


if __name__ == "__main__":
    unittest.main()
