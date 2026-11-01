"""Pure encoder capabilities and validated FFmpeg encoding options.

Hardware encoders intentionally expose only bitrate control: their CQ/QP modes
are not software CRF. Capabilities describe known options, not GPU availability.
"""

from dataclasses import dataclass, replace
from math import isfinite


@dataclass(frozen=True)
class EncoderCapabilities:
    presets: tuple[str, ...]
    rate_controls: tuple[str, ...]
    preset_flag: str | None


_SOFTWARE = EncoderCapabilities(
    ("Default", "ultrafast", "superfast", "veryfast", "faster", "fast",
     "medium", "slow", "slower", "veryslow", "placebo"),
    ("Bitrate", "CRF"),
    "-preset",
)
_NVENC = EncoderCapabilities(
    ("Default", "p1", "p2", "p3", "p4", "p5", "p6", "p7"),
    ("Bitrate",),
    "-preset",
)
_AMF = EncoderCapabilities(
    ("Default", "speed", "balanced", "quality"),
    ("Bitrate",),
    "-quality",
)
_QSV = EncoderCapabilities(
    ("Default", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"),
    ("Bitrate",),
    "-preset",
)
_ENCODERS = {
    "x264": _SOFTWARE,
    "libx264": _SOFTWARE,
    "x265": _SOFTWARE,
    "libx265": _SOFTWARE,
    "h264_nvenc": _NVENC,
    "hevc_nvenc": _NVENC,
    "h264_amf": _AMF,
    "hevc_amf": _AMF,
    "h264_qsv": _QSV,
    "hevc_qsv": _QSV,
}
_UNKNOWN = EncoderCapabilities(("Default",), ("Bitrate",), None)


def encoder_capabilities(codec: str) -> EncoderCapabilities:
    """Return known choices without importing or probing FFmpeg.

    Unknown codecs keep the legacy default bitrate path, allowing FFmpeg's x264
    fallback, but cannot receive codec-specific presets or software CRF flags.
    """
    return _ENCODERS.get(codec, _UNKNOWN)


@dataclass(frozen=True)
class EncoderOptions:
    preset: str = "Default"
    rate_control: str = "Bitrate"
    crf: float = 23.0
    bitrate_kbps: int = 50000

    def validate(self, codec: str) -> None:
        """Reject unsupported choices and invalid values in the active mode."""
        capabilities = encoder_capabilities(codec)
        if self.preset not in capabilities.presets:
            raise ValueError(f"Unsupported preset {self.preset!r} for encoder {codec!r}")
        if self.rate_control not in capabilities.rate_controls:
            raise ValueError(
                f"Unsupported rate control {self.rate_control!r} for encoder {codec!r}"
            )
        if self.rate_control == "CRF":
            if (isinstance(self.crf, bool)
                    or not isinstance(self.crf, (int, float))
                    or not 0 <= self.crf <= 51
                    or not isfinite(self.crf)):
                raise ValueError("CRF must be a finite number from 0 to 51")
        elif (isinstance(self.bitrate_kbps, bool)
              or not isinstance(self.bitrate_kbps, int)
              or self.bitrate_kbps <= 0):
            raise ValueError("Bitrate must be a positive whole number of kb/s")

    def arguments(self, codec: str) -> list[str]:
        """Return only this encoder's preset and active rate-control flags."""
        self.validate(codec)
        capabilities = encoder_capabilities(codec)
        arguments = []
        if self.preset != "Default":
            arguments += [capabilities.preset_flag, self.preset]
        if self.rate_control == "CRF":
            arguments += ["-crf", str(self.crf)]
        else:
            arguments += ["-b:v", f"{self.bitrate_kbps}k"]
        return arguments


def normalize_encoder_options(codec: str, options: EncoderOptions) -> EncoderOptions:
    """Adapt choices to a new codec without masking invalid numeric values."""
    capabilities = encoder_capabilities(codec)
    return replace(
        options,
        preset=options.preset if options.preset in capabilities.presets else "Default",
        rate_control=(options.rate_control
                      if options.rate_control in capabilities.rate_controls else "Bitrate"),
    )
