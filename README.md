<html>
<body>
    <div align="center">
        <img src="https://github.com/Djdefrag/QualityScaler/blob/main/Assets/logo.png" width="175">
        <br><br> QualityScaler - image/video AI upscaler app <br><br>
        <a href="https://jangystudio.itch.io/qualityscaler">
            <button>
                <img src="https://static.itch.io/images/badge-color.svg" width="225" height="70">
            </button>
        </a>
        <a href="https://store.steampowered.com/app/2463110/QualityScaler/">
            <button>
                 <img src="https://images.squarespace-cdn.com/content/v1/5b45fae8b98a78d9d80b9c5c/1531959264455-E7B8MJ3VMPX0593VGCZG/button-steam-available-fixed-2.png" width="250" height="70">
            </button>
        </a>
    </div>
    <br>
    <div align="center">
        <img src="https://github.com/user-attachments/assets/040e3880-3adb-478a-9a82-df2c5dc12fad">
    </div>
</body>
</html>

<div align="center">

[![Release](https://img.shields.io/github/v/release/Djdefrag/QualityScaler?style=flat-square&color=blue)](https://github.com/Djdefrag/QualityScaler/releases)
[![Downloads](https://img.shields.io/github/downloads/Djdefrag/QualityScaler/total?style=flat-square&color=success)](https://github.com/Djdefrag/QualityScaler/releases)
[![License](https://img.shields.io/github/license/Djdefrag/QualityScaler?style=flat-square)](https://github.com/Djdefrag/QualityScaler/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/Djdefrag/QualityScaler?style=flat-square&color=yellow)](https://github.com/Djdefrag/QualityScaler/stargazers)

</div>

## 📑 Table of Contents
- [What is QualityScaler?](#-what-is-qualityscaler)
- [Features](#-features)
- [How to use QualityScaler?](#-how-to-use-qualityscaler)
- [Other AI projects](#-other-ai-projects)
- [Credits](#-credits)
- [Citations](#-citations)
- [How is made](#-how-is-made)
- [Make it work by yourself](#-make-it-work-by-yourself-)
- [Portable Windows executable](#portable-windows-executable)
- [Video frame cache and resume](#video-frame-cache-and-resume)
- [Backend layout and tests](#backend-layout-and-tests)
- [Requirements](#-requirements)
- [Roadmap](#-roadmap)
- [Examples](#-examples)

## ❓ What is QualityScaler?
QualityScaler is a Windows app powered by AI to enhance, upscale and de-noise photographs and videos.

This source tree combines the 2026.4 interface and inference engine with
single-database video caching, resumable jobs, simple output filenames, and
native file drag-and-drop.

## ✨ Features
- Elegant and easy to use GUI, with app zoom from 50% to 175%
- Image and video upscale
- Automatic GPU detection by name, with dedicated VRAM information
- DirectML inference when available, with CPU fallback when DirectML is unavailable
- Compatible images - jpg, png, tif, bmp, webp, heic
- Compatible video - mp4, wmv, webm, mkv, flv, gif, avi, mov, mpg, qt, 3gp
- Automatic overlapping, feathered image tiling to reduce seams and limit VRAM use
- Resize image/video before upscaling and scale the output resolution
- AI sharpening: OFF, Low, or High (the 2026 control replaces the old blending UI)
- Upscale preview window with video frame seeking, scroll zoom, and drag-to-pan
- Video upscaling STOP & RESUME, with frames and job settings in one SQLite cache
- Native file drag-and-drop and appendable input selection
- 🔒 Privacy focused - no internet connection required, everything runs on your PC

## 🎬 How to use QualityScaler?
[movie_max_vp9.webm](https://github.com/user-attachments/assets/4a1984be-c8c8-464b-a3d1-43445635b165)

### Select input files

Drag images or videos from File Explorer onto the **left input pane**, or use
**SELECT FILES**. Drop additional files onto the populated list or use **ADD FILES**
to append them; duplicates are ignored. **CLEAN** clears the selection so you can
start again.

Multiple files, spaces, Unicode and braces in filenames are supported. Folders,
unsupported extensions and unreadable media are skipped without clearing existing
inputs. Input changes are disabled while a job is running. Files are never moved
by a drop.

### 2026 processing controls

Choose one of the nine standard models, input/output resolution percentages,
and an **AI sharpening** level. The GPU menu lists detected adapters by name;
VRAM and a suitable hardware encoder are selected from the detected GPU.
Keep the VRAM limit within the selected adapter's capacity. CPU inference is
available when the runtime has no DirectML provider, but is substantially slower;
CPU inference does not make an unavailable hardware video encoder usable.

**App zoom** changes the interface scale, not the output resolution. Large inputs
are processed with overlapping tiles and feathered edges. These inference and
sharpening rules differ from 4.7, so old caches must not be reused as new jobs.

The desktop window can be resized or maximized. Settings scroll on shorter
windows, and the full-width bottom status bar wraps long filenames and progress
messages. Very long messages scroll within the bar without covering Stop.
Video jobs show separate extraction and AI upscaling status/progress rows.
Frame totals are estimates until extraction finishes; unknown totals show saved
frame counts instead of a percentage.
Scale edits refresh file previews after typing pauses instead of
rebuilding the cards on every keystroke.

### Preview an upscale

Select input files, then click **Preview upscale ↗** in the settings pane. Choose
an input in the preview window and move the seek bar (or use the frame arrows) to
upscale a video frame. Still images are previewed with seeking disabled. The preview
uses the current AI model, GPU/VRAM limit, input/output scales, and sharpening;
setting changes refresh it automatically. **Refresh** retries the current frame.

Scroll over the result to zoom around the pointer and drag to pan. **Fit** and
**100%** reset the view; zoom only changes the display, never the inference size.
The preview runs in a separate worker process and does not save output files or
modify video caches. It shows the result before video/image encoding, so final
codec compression can look different. Starting a batch or changing the input
list closes the preview to release its GPU memory.

After the last output is written, the app shows **Finalizing** while workers
release their resources. Completion is reported only after the worker has exited
successfully and the file cards are refreshed. Stop and close remain responsive
while progress is checkpointed; a hung worker still has the 10-second hard-stop
fallback.

## 🤖 Other AI projects
- [RealScaler](https://github.com/Djdefrag/RealScaler) - image/video AI upscaler (Real-ESRGAN)
- [FluidFrames.RIFE](https://github.com/Djdefrag/FluidFrames.RIFE) - video AI frame generation

## 🙏 Credits
- [BSRGAN](https://github.com/cszn/BSRGAN)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [IRCNN](https://github.com/lipengFu/IRCNN)

## ❤️ Citations
- [80.lv - Great AI-powered tools for upscaling images](https://80.lv/articles/80-level-digest-great-ai-powered-tools-for-upscaling-images/)
- [TimeSaverVFX - AI Upscale](https://timesavervfx.com/ai-upscale/)

## 🛠️ How is made
QualityScaler is completely written in Python, from backend to frontend.
- [PyTorch](https://github.com/pytorch/pytorch)
- [ONNX](https://github.com/onnx/onnx)
- [onnxconverter-common](https://github.com/microsoft/onnxconverter-common)
- [onnxruntime-directml](https://github.com/microsoft/onnxruntime)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- [OpenCV](https://github.com/opencv/opencv)
- [PyInstaller](https://github.com/pyinstaller/pyinstaller)

## 🧑‍💻 Make it work by yourself
**Prerequisites**
- [Python 3.11 x64](https://www.python.org/downloads/) installed on your PC
- [VSCode](https://code.visualstudio.com/) installed on your PC (optional editor)
- The current [AI model bundle](https://gofile.io/d/b4Ds9u) downloaded
- [ffmpeg.exe](https://www.gyan.dev/ffmpeg/builds/) downloaded (release build > ffmpeg-release-essentials.7z)

**Getting started**
- Download the project on your PC (green button Code > Download ZIP)
- Extract the project from the .zip
- Extract the AI models files in the `/AI-onnx` folder
- Extract `ffmpeg.exe` in the `/Assets` folder
- Open the project with VSCode (drag & drop the project directory onto VSCode)
- Click on `QualityScaler.py` in the left sidebar (VSCode will ask to install the Python extension)
- Install dependencies in the Python environment you will use: `python -m pip install -r requirements.txt`
- Close VSCode and re-open it (this refreshes all the installed dependencies)
- Run `python QualityScaler.py`, or click the "Play" button in the upper right corner of VSCode

The standard 2026 model catalog requires these exact filenames in `AI-onnx/`:

```text
LVAx2_fp16.onnx
RealESR_Gx4_fp16.onnx
RealESR_Ax4_fp16.onnx
BSRGANx2_fp16.onnx
BSRGANx4_fp16.onnx
RealESRGANx4_fp16.onnx
MSharpx4_fp16.onnx
IRCNN_Mx1_fp16.onnx
IRCNN_Lx1_fp16.onnx
```

Obtain the current bundle if the files are missing; if the download service is
unavailable, request the bundle from its maintainer. The old
`RealESR_Animex4_fp16.onnx` is **not** an alias for `RealESR_Ax4_fp16.onnx`:
do not rename or substitute it. Model binaries are ignored by Git.

ExifTool is not required or bundled. Supported source video metadata is mapped
by FFmpeg (`-map_metadata`), subject to the chosen output container; there is
no separate ExifTool pass for image or video metadata.

## Portable Windows executable

The portable build produces `dist/QualityScaler.exe`, a single Windows x64
executable containing Python, the DirectML runtime, FFmpeg, UI assets, native
TkDND support, and **all nine standard models** listed above. A completed build
needs no Python installation or FFmpeg PATH configuration on the target PC.
Each launch extracts its bundled resources to the Windows temporary directory.
PyInstaller 6.16.0's native bootloader splash shows the logo and the file currently
being unpacked, before Python or the desktop can run. This is per-file progress,
not a percentage; unpacking a large model can leave the same filename visible
for a while. After extraction, the status line switches to short startup phases
and stays until the desktop's first `window.update()` has completed. Startup
errors close the splash before propagating.

The splash reuses `Assets/logo.png`, with dark status text in the lower margin;
no separate image or second Python/Tk root is created. The lazy
`qualityscaler.startup.update_splash(text)` and `close_splash()` helpers are
enabled only by the frozen desktop launcher after `freeze_support()`. Source
runs, CLI/self-test UI calls, and multiprocessing workers cannot attach to or
close another process's splash.

An existing 4.7 executable does not contain this 2026.4 forward-port. Rebuilding
requires the current model bundle; these instructions do not claim that a new
2026 executable has been built, published, or tested.

To rebuild locally, install Python 3.11 x64 and provide `Assets/ffmpeg.exe`,
the repository's PNG/ICO assets, and every standard `*_fp16.onnx` model.
Run from PowerShell:

```powershell
.\build.ps1
# Reuse an already-provisioned .venv without installing dependencies again:
.\build.ps1 -SkipInstall
```

The script creates `.venv` when needed, installs `requirements.txt`, and uses
`QualityScaler.spec` to build the executable. The requirements pin
`onnxruntime-directml==1.24.4` and `pyinstaller==6.16.0`; the packager stays at
the established version while the inference runtime follows 2026.4.
`-SkipInstall` skips pip operations, not resource checks.

Before PyInstaller analysis, the spec rejects missing standard models, FFmpeg,
app assets, licenses, or required DirectML runtime DLLs. It does **not** silently
produce an incomplete bundle when only older models are available. Additional
local `*_fp16.onnx` files are bundled too, but do not satisfy a missing standard
model. TkDND's supported PyInstaller hook preserves its native DLL and Tcl
package paths.

After a successful build the script prints the executable's path, size, and
SHA-256. `build/`, `dist/`, model binaries, and FFmpeg are ignored by Git.
The build is unsigned; no installer, administrator access, or system-wide
installation is needed.

### Verify a newly built executable

After obtaining all current resources and rebuilding, run the **actual
executable** to check bundled model/DLL discovery, native file-drop dispatch,
DirectML kernel execution, spawned GPU workers, SQLite interruption/resume,
FFmpeg encoding, and preserved audio:

```powershell
$report = Join-Path $PWD 'build\packaged-self-test.json'
# The native bootloader runs before --self-test can be parsed. Suppress its
# splash before launch; self-test deliberately never attaches to splash IPC.
$previousSplashSuppression = $env:PYINSTALLER_SUPPRESS_SPLASH_SCREEN
try {
    $env:PYINSTALLER_SUPPRESS_SPLASH_SCREEN = '1'
    $process = Start-Process .\dist\QualityScaler.exe -ArgumentList ('--self-test "' + $report + '"') -Wait -PassThru
} finally {
    $env:PYINSTALLER_SUPPRESS_SPLASH_SCREEN = $previousSplashSuppression
}
$process.ExitCode
Get-Content $report
```

This unattended mode writes a JSON result and sibling `.log`, returns nonzero
on failure, and removes its temporary test media. It briefly constructs and
closes the desktop UI without changing saved user preferences, then tests
DirectML inference and video resume. It requires a working desktop, DirectML
hardware/runtime, and the current bundled models. Read the generated report;
the command being documented is not evidence of a passing packaged build.

Validate the splash separately with a normal desktop launch, with
`PYINSTALLER_SUPPRESS_SPLASH_SCREEN` unset: it should be visible during onefile
extraction, show import/GPU/UI preparation status, then disappear when the
desktop is ready, without reappearing as workers start. PyInstaller's bootloader
also accepts `PYINSTALLER_SUPPRESS_SPLASH_SCREEN=1` for deliberately splash-free
desktop launches. Unit tests mock the native target/IPC; they do not prove that
a packaged splash was displayed or that desktop readiness was reached.

## Video frame cache and resume

Video jobs store original and upscaled JPEG frames in **one SQLite database**,
not individual frame files. The database also records the full job settings,
source identity, frame indices, extraction completion, current stage, and errors.
Output names are simply `clip_upscaled.mp4` (or the chosen extension); image
outputs similarly use `photo_upscaled.png`. Settings are not encoded in filenames.

The cache lives beside the output in `clip.mp4.qualityscaler/frames.sqlite3`.
Choose **Save frames: ON** to retain it after a successful encode. With this option
OFF, only a successful job removes the cache. Stopped and failed jobs keep it.
SQLite may create a temporary rollback journal while a batch is being saved;
the number of files stays constant regardless of video length.

**Cache compatibility:** new jobs use `job_version = 3`. FFmpeg applies the
selected input scale with area interpolation **before JPEG compression and
SQLite storage**. The cache records the original dimensions, scaled input
dimensions, and `source_frame_layout = "ffmpeg-area-v1"`. Workers do not apply
the input scale a second time, and final output geometry is still derived from
the original video. JPEG/interpolation order changes, so pixels need not be
bit-identical to the previous full-resolution-cache path.

Existing 2026.4 `job_version = 2` jobs remain resumable using their original
full-resolution source frames and worker-side scaling, even if extraction was
interrupted. They are not converted or mixed with pre-scaled frames. Older
executables reject new version-3 jobs rather than scaling them twice.
The earlier 4.7 SQLite jobs (`job_version = 1`) are rejected rather than reused with different
sharpening, tiling, or model semantics. Archive the old cache, choose a different
output folder, or finish it with the matching older version. There is no automatic
cache conversion.

- Start the same video again with the same AI, resize, tile and sharpening settings
  to resume in the GUI. Finished extraction is reused, and only missing upscaled
  frames are processed.
- A partial extraction is replayed from the beginning, reusing saved frames.
  This avoids inaccurate timestamp/keyframe seeking. AI upscaling consumes
  available cached frames while extraction continues, including during replay.
  Encoding starts only after extraction succeeds and every upscaled frame is saved.
- Frame writes are committed in batches of 16. A graceful stop commits the
  current batch; a forced termination can lose the uncommitted batch, but cannot
  mark those frames complete. SQLite recovers interrupted transactions on reopen.
- Already-dispatched inference work must finish before a graceful stop can
  finish. Previously saved results are committed before waiting. The desktop
  stop handler has a bounded wait and can force termination if inference hangs.
- A changed source (path, size or modification time), model weights (SHA-256), or
  processing settings is rejected rather than mixed into an existing job. Restore the saved
  settings, choose a different output folder, or move the old cache aside to
  start fresh. GPU, worker count and encoding codec can change on resume.
- For a processing-settings mismatch, the **Upscale error** dialog also offers
  **Clear cache and continue**. It discards only that video's cached source/AI
  frames and restarts it using the **current UI settings** by default. Select
  the unchecked **Use settings from failed run** option to reuse that run's
  choices instead. The batch continues without repeating earlier files.
  Choosing a different output folder keeps the old cache and never authorizes
  clearing a cache in the new folder; another conflict there requires a new
  confirmation. Other caches are untouched, and any existing output is replaced
  only after encoding succeeds. **Cancel** or closing the dialog leaves the
  cache unchanged; cancel first if you want to edit the current settings.
- Encoding streams cached frames in frame-index order directly to FFmpeg, with
  optional source audio and supported video metadata. Failed encoding retains
  the frames. The final output is replaced only after FFmpeg succeeds.
- WMV inputs (including uppercase `.WMV`) use the same picker, drag-and-drop,
  upscaling and resume pipeline. Exports remain MP4/MKV/AVI/MOV. Audio is copied
  when the output container supports it; incompatible source audio, such as
  WMA in MP4, is retried as AAC rather than discarded.
- Only one runner may own a cache at a time. A crash-released OS lock prevents
  simultaneous GUI/CLI resumes from mixing job state. Its tiny, stable lock file
  lives in the OS temporary directory, not among the cached frames.
- The status bar keeps only the latest status for each active task, on separate
  lines. Each task has its own remaining time (for example, `1d 2h 3m 4s`) and ETA
  in the system's local date/time and timezone; estimates are not combined.
  A task shows `estimating...` until it has a known total and measured progress.
  These are stage estimates, not predictions for unstarted stages or later files
  in the batch. Resumed inference measures only newly processed frames, and
  encoding retries reset their estimate.

Inspect the saved settings/progress, or resume directly using the recorded
configuration without re-entering it in the GUI:

```sh
python -m qualityscaler inspect "path/to/clip.mp4.qualityscaler/frames.sqlite3"
python -m qualityscaler resume "path/to/clip.mp4.qualityscaler/frames.sqlite3"
python -m qualityscaler resume "path/to/clip.mp4.qualityscaler/frames.sqlite3" --codec x264
```

Press Ctrl+C in the resume command to checkpoint and pause at the current
inference boundary. Resume requires the original video and AI model files to
remain available. Old per-frame JPEG folders and the experimental
`framevault.mpeg` are not automatically imported or deleted.
Existing simple-named outputs are replaced after successful processing, so use a
different output folder when keeping multiple model/settings variants.

## Backend layout and tests

`QualityScaler.py` is a spawn-safe desktop launcher; desktop layout and controls
live in `qualityscaler/gui.py`. GUI-independent backend modules are in
`qualityscaler/`: `frame_store.py` owns SQLite transactions, `jpeg_stream.py`
parses streamed JPEGs, `ffmpeg.py` handles extraction/encoding and cancellation,
`video.py` owns resumable jobs and bounded spawned inference workers, `ai.py`
owns model inference, and `media.py`, `paths.py`, `jobs.py` provide shared
helpers/batch jobs.

Each inference process owns its own ONNX Runtime session; SQLite stays in the
parent. Process isolation avoids the Python DirectML allocator crashes seen
when independent sessions run concurrently in threads. Only encoded JPEG bytes
cross process boundaries, with at most twice the worker count in flight.
Extraction runs in a background thread with a separately bounded JPEG queue;
the parent remains the sole SQLite writer for both source and upscaled frames.

Run the tests with the same Python environment used for the application:

```sh
python -m unittest discover -s tests -v
```

Storage/parser and portable-spec tests need only Python's standard library.
Media and pipeline tests use NumPy and OpenCV; FFmpeg integration tests also need
`ffmpeg` on PATH (otherwise those tests are skipped). Ordinary pipeline tests use
a deterministic resize engine, not downloaded ONNX models or DirectML hardware.
The portable-spec tests use mocked PyInstaller hooks and fixture model files:

```powershell
py -3.11 -m unittest discover -s tests -p test_build.py -v
py -3.11 -m unittest discover -s tests -p test_startup.py -v
```

These are commands for local validation, not claims that the full suite or
a 2026 packaged executable has passed on every supported configuration.

### Real-model DirectML checks (opt-in)

Place a current model such as `RealESR_Gx4_fp16.onnx` in `AI-onnx/`, using the
model bundle linked in the setup guide above. Model binaries are ignored by Git
and should not be committed.

To run the real-model tests in PowerShell:

```powershell
$env:QUALITYSCALER_TEST_MODEL = "RealESR_Gx4"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_real_model.py -v
Remove-Item Env:\QUALITYSCALER_TEST_MODEL
```

These opt-in tests require DirectML hardware/runtime and FFmpeg on PATH. They
check actual GPU execution through ONNX profiling, then interrupt and resume a
real upscaling job through the SQLite cache, checking that only missing frames
are inferred and that the encoded output retains its frame count, resolution,
frame rate, and audio. They are skipped unless `QUALITYSCALER_TEST_MODEL` is set;
CPU fallback is not a substitute for verifying DirectML execution.

## 💻 Requirements
| Requirement | Minimum |
|---|---|
| OS | Windows 10 / Windows 11 |
| RAM | 8 GB or more |
| GPU | DirectX 12 compatible GPU with ≥ 4 GB VRAM recommended for DirectML |
| CPU fallback | Available when DirectML is unavailable; slower than GPU inference |

## 🗺️ Roadmap

<details>
<summary>Click to expand full changelog</summary>

- [x] 1.X versions
    - [x] Switch to Pytorch-directml to support all Directx12 compatible gpu (AMD, Intel, Nvidia)
    - [x] New GUI with Windows 11 style
    - [x] Include audio for upscaled video
    - [x] Optimizing video frame resize and extraction speed
    - [x] Multi GPU support (for pc with double GPU, integrated + dedicated)
    - [x] Python 3.10 (expecting ~10% more performance)
- [x] 2.X versions
    - [x] New, completely redesigned graphical interface based on @customtkinter
    - [x] Upscaling images and videos at once (currently it is possible to upscale images or single video)
    - [x] Upscale multiple videos at once
    - [x] Choose upscaled video extension
    - [x] Interpolation between the original and upscaled image/video
    - [x] More Interpolation levels (Low, Medium, High)
    - [x] Show the remaining time to complete video upscaling
    - [x] Support for SRVGGNetCompact AI architecture
    - [x] Metadata extraction and application from original file to upscaled file (via exiftool)
- [x] 3.X versions
    - [x] New AI engine powered by onnxruntime-directml (https://github.com/microsoft/onnxruntime))
    - [x] Python 3.11 (performance improvements)
    - [x] Python 3.12 (performance improvements)
    - [x] Display images/videos upscaled resolution in the GUI
    - [x] Updated FFMPEG to version 7.x (latest release)
    - [x] Saving user settings (AI model, GPU, CPU  etc.)
    - [x] Video AI multi-threading upscale
    - [x] Video upscaling STOP&RESUME
- [x] 4.X version
    - [x] Hardware accelerated video encoding (nvenc / amf / qsv)
    - [x] More video extensions (.mp4 / .mkv / .avi / .mov)
    - [x] Choose output resolution scaling
    - [x] Video AI multi-threading improvements
    - [x] Updated AI-engine (onnxruntime-directml)
    - [x] Video frames extraction by FFMPEG (x10 times faster)
- [ ] 2026.X version
    - [x] Implement new AI models
    - [x] App interface % scaling
    - [ ] TTA upscale mode

</details>

The changelog above includes historical features: the current 2026 interface
uses sharpening rather than the old interpolation/blending control, and the
current metadata path does not use ExifTool.

## 🖼️ Examples

#### Videos
![original](https://user-images.githubusercontent.com/32263112/209139620-bdd028f8-d5fc-40de-8f3d-6b80a14f8aab.gif)

https://user-images.githubusercontent.com/32263112/209139639-2b123b83-ac6e-4681-b94a-954ed0aea78c.mp4

#### Images
![test](https://user-images.githubusercontent.com/32263112/166690007-f1601487-7b94-4f2c-b4e2-436bc189a26e.png)

![ORIGINAL](https://user-images.githubusercontent.com/32263112/226847190-e4dbda21-8896-456d-8120-3137f3d2ac62.png)

![Bsrgan x4](https://user-images.githubusercontent.com/32263112/168884625-c869baee-4cca-4a33-bdad-b65d9c29889d.png)

![Bsrgan x4 (2)](https://user-images.githubusercontent.com/32263112/197983965-40785dbd-78c6-48a0-a1eb-39d9c3278f42.png)

![Bsrgan x4 (3)](https://user-images.githubusercontent.com/32263112/197983979-5857a855-d402-4fab-9217-ee5bd057bd01.png)

![Bsrgan x4](https://user-images.githubusercontent.com/32263112/198290909-277e176e-ccb4-4a4b-8531-b182a18d566a.png)
