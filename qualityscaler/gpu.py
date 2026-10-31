"""GPU discovery and ONNX Runtime provider information, independent of the GUI."""

from dataclasses import dataclass
import logging
import sys
from typing import ClassVar, Optional


logger = logging.getLogger(__name__)


@dataclass
class GPU:
    """A DXGI hardware adapter and the registry used by GPU selection controls."""

    name: str
    vram_gb: float
    device_id: int
    vendor_id: int = 0

    AUTO: ClassVar[str] = "Auto"
    NO_GPU: ClassVar[str] = "No GPU found"
    VENDOR_NVIDIA: ClassVar[int] = 0x10DE
    VENDOR_AMD: ClassVar[int] = 0x1002
    VENDOR_INTEL: ClassVar[int] = 0x8086
    detected: ClassVar[list["GPU"]] = []

    @property
    def vram_limit(self) -> int:
        """Dedicated VRAM floored to whole GB, never below 1."""
        return max(1, int(self.vram_gb))

    @property
    def hardware_codec(self) -> Optional[str]:
        return {
            self.VENDOR_NVIDIA: "h264_nvenc",
            self.VENDOR_AMD: "h264_amf",
            self.VENDOR_INTEL: "h264_qsv",
        }.get(self.vendor_id)

    @classmethod
    def detect(cls) -> None:
        cls.detected = cls._enumerate()

    @classmethod
    def find(cls, name: str) -> Optional["GPU"]:
        return next((gpu for gpu in cls.detected if gpu.name == name), None)

    @classmethod
    def menu_list(cls) -> list[str]:
        return [gpu.name for gpu in cls.detected] if cls.detected else [cls.NO_GPU]

    @classmethod
    def default(cls) -> str:
        return cls.detected[0].name if cls.detected else cls.NO_GPU

    @classmethod
    def device_id_for(cls, selected: str) -> str:
        """Resolve a display name, using the first adapter or Auto as fallback."""
        if not cls.detected:
            return cls.AUTO
        gpu = cls.find(selected)
        return str(gpu.device_id) if gpu is not None else str(cls.detected[0].device_id)

    @classmethod
    def vram_for(cls, selected: str) -> Optional[int]:
        gpu = cls.find(selected)
        return gpu.vram_limit if gpu is not None else None

    @classmethod
    def codec_for(cls, selected: str) -> Optional[str]:
        gpu = cls.find(selected)
        return gpu.hardware_codec if gpu is not None else None

    @staticmethod
    def _shorten_name(raw_name: str) -> str:
        """Remove vendor/brand filler without reducing integrated GPUs to 'Graphics'."""
        if not raw_name:
            return raw_name
        cleaned = raw_name
        for marker in ("(R)", "(TM)", "(C)", "\u00ae", "\u2122", "\u00a9"):
            cleaned = cleaned.replace(marker, " ")

        def strip_words(text: str, words: set[str]) -> str:
            return " ".join(token for token in text.split() if token.upper() not in words)

        vendors = {
            "NVIDIA", "GEFORCE", "AMD", "INTEL", "CORPORATION",
            "ADVANCED", "MICRO", "DEVICES",
        }
        primary = strip_words(cleaned, vendors | {"RADEON"})
        generic_words = {"GRAPHICS", "SERIES", "FAMILY"}
        if not primary or all(token.upper() in generic_words for token in primary.split()):
            primary = strip_words(cleaned, vendors)
        return primary if primary else raw_name.strip()

    @staticmethod
    def _enumerate() -> list["GPU"]:
        """Enumerate DXGI hardware adapters without vendor SDKs or an ORT session."""
        if sys.platform != "win32":
            return []

        import ctypes

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_uint32),
                ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        class DXGI_ADAPTER_DESC1(ctypes.Structure):
            _fields_ = [
                ("Description", ctypes.c_wchar * 128),
                ("VendorId", ctypes.c_uint),
                ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint),
                ("Revision", ctypes.c_uint),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", LUID),
                ("Flags", ctypes.c_uint),
            ]

        software_adapter = 2
        factory_iid = GUID(
            0x770AAE78, 0xF26F, 0x4DBA,
            (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
        )

        def com_method(obj_ptr, index, restype, argtypes):
            vtable = ctypes.cast(obj_ptr, ctypes.POINTER(ctypes.c_void_p))[0]
            func_addr = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[index]
            return ctypes.WINFUNCTYPE(restype, *argtypes)(func_addr)

        gpus = []
        seen_names = {}
        try:
            factory = ctypes.c_void_p()
            hr = ctypes.windll.dxgi.CreateDXGIFactory1(
                ctypes.byref(factory_iid), ctypes.byref(factory),
            )
            if hr != 0 or not factory.value:
                return []

            # IDXGIFactory1::EnumAdapters1 = 12; IUnknown::Release = 2.
            enum_adapters = com_method(
                factory, 12, ctypes.c_int32,
                [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
            )
            factory_release = com_method(factory, 2, ctypes.c_ulong, [ctypes.c_void_p])
            try:
                adapter_index = 0
                hardware_index = 0
                while True:
                    adapter = ctypes.c_void_p()
                    if enum_adapters(factory, adapter_index, ctypes.byref(adapter)) != 0:
                        break
                    # IDXGIAdapter1::GetDesc1 = 10.
                    adapter_release = com_method(adapter, 2, ctypes.c_ulong, [ctypes.c_void_p])
                    try:
                        get_desc = com_method(
                            adapter, 10, ctypes.c_int32,
                            [ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1)],
                        )
                        desc = DXGI_ADAPTER_DESC1()
                        if get_desc(adapter, ctypes.byref(desc)) != 0:
                            # A missing description makes device-ID mapping unsafe.
                            raise RuntimeError("DXGI could not describe an adapter")
                        if not desc.Flags & software_adapter:
                            display_name = GPU._shorten_name(desc.Description)
                            seen_names[display_name] = seen_names.get(display_name, 0) + 1
                            if seen_names[display_name] > 1:
                                display_name = f"{display_name} ({seen_names[display_name]})"
                            gpus.append(GPU(
                                name=display_name,
                                vram_gb=round(desc.DedicatedVideoMemory / (1024 ** 3), 1),
                                device_id=hardware_index,
                                vendor_id=desc.VendorId,
                            ))
                            hardware_index += 1
                    finally:
                        adapter_release(adapter)
                    adapter_index += 1
            finally:
                factory_release(factory)
        except Exception:
            logger.warning("QualityScaler GPU detection failed", exc_info=True)
            return []
        return gpus


def get_available_AI_providers() -> list[str]:
    """Return providers in the installed ORT build, importing only on request."""
    try:
        from onnxruntime import get_available_providers
        return list(get_available_providers())
    except Exception:
        return []


def is_directml_available() -> bool:
    return any(
        "Dml" in provider or "DirectML" in provider
        for provider in get_available_AI_providers()
    )


def get_AI_providers() -> list[str]:
    """Prefer DirectML, with CPU inference when the build has no DirectML."""
    return ["DmlExecutionProvider"] if is_directml_available() else ["CPUExecutionProvider"]


def get_AI_engine_info() -> str:
    """A human-readable runtime label, or an empty label if ORT is unavailable."""
    try:
        from onnxruntime import get_version_string
        version = get_version_string()
        provider = "DirectML" if is_directml_available() else "CPU"
        return f"AI engine {version} + {provider}"
    except Exception:
        return ""
