# Build with Python 3.11 / PyInstaller 6.16.0; see build.ps1.
# Keep all resource paths relative to the extracted onefile bundle, not the exe.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, get_package_paths


ROOT = Path(SPECPATH).resolve()
ASSETS = ROOT / "Assets"
MODELS = ROOT / "AI-onnx"
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

# Check before analysis so a missing local download cannot produce a broken exe.
# Every model offered by the normal menu must work in the portable executable.
required_files = (
    ROOT / "QualityScaler.py",
    ASSETS / "logo.ico",
    ASSETS / "logo.png",
    ASSETS / "ffmpeg.exe",
    ROOT / "LICENSE",
    ROOT / "LICENSE.txt",
) + tuple(MODELS / f"{name}_fp16.onnx" for name in STANDARD_MODELS)
missing_files = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]
if missing_files:
    raise SystemExit(
        "Cannot build QualityScaler; required local files are missing:\n  "
        + "\n  ".join(missing_files)
        + "\nProvision the assets and models before running build.ps1."
    )

# External tools are resource files at sys._MEIPASS/Assets, alongside the images,
# so the application's resource lookups work without relying on the user's PATH.
datas = [
    (str(path), "Assets")
    for pattern in ("*.png", "*.ico")
    for path in sorted(ASSETS.glob(pattern))
    if path.is_file()
]
datas += [(str(ASSETS / "ffmpeg.exe"), "Assets")]
datas += [
    (str(path), "AI-onnx")
    for path in sorted(MODELS.glob("*_fp16.onnx"))
    if path.is_file()
]
datas += [(str(ROOT / name), ".") for name in ("LICENSE", "LICENSE.txt")]
datas += sorted(collect_data_files("customtkinter"))

# Preserve the helper's package-relative DLL destinations (not the bundle root).
binaries = sorted(collect_dynamic_libs("onnxruntime"))
required_dlls = {"directml.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll"}
missing_dlls = required_dlls - {Path(source).name.lower() for source, _ in binaries}
if missing_dlls:
    raise SystemExit(
        "onnxruntime-directml is missing required DLLs: "
        + ", ".join(sorted(missing_dlls))
        + ". Install requirements.txt into .venv before building."
    )

# Use tkinterdnd2's supported hook to collect pkgIndex.tcl, TkDND scripts and the
# native library for this Python's OS, CPU and Tcl version, preserving package
# paths. Discover the hook without importing tkinterdnd2 (and therefore Tk).
tkdnd_hook_dir = Path(get_package_paths("tkinterdnd2")[1]) / "__pyinstaller"

# Analyze the spawn-safe launcher without importing the application here.
# Standard hooks handle multiprocessing, NumPy, OpenCV, Pillow and Tk.
a = Analysis(
    [str(ROOT / "QualityScaler.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[str(tkdnd_hook_dir)],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

# Native bootloader splash: visible while the onefile archive is still unpacking.
# text_pos enables PyInstaller's per-file extraction progress; Python takes over
# the same status line after freeze_support(), then closes it when the UI is ready.
# Reuse the logo, with dark text in its clear lower margin; no generated artwork.
splash = Splash(
    str(ASSETS / "logo.png"),
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(16, 472),
    # Pixel sizing keeps this line inside the logo's lower margin at high DPI.
    text_size=-14,
    text_font="TkDefaultFont",
    text_color="#151515",
    text_default="Starting QualityScaler...",
    max_img_size=(480, 480),
    full_tk=False,
    always_on_top=False,
)

# Supplying binaries/data directly to EXE, with no COLLECT, makes this onefile.
# Both Splash outputs are required: the native resource and its Tcl/Tk binaries.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,
    splash.binaries,
    [],
    name="QualityScaler",
    console=False,
    icon=str(ASSETS / "logo.ico"),
    debug=False,
    strip=False,
    upx=False,
)
