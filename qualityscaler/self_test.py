"""Unattended verification of the actual portable executable and GPU workers."""

from contextlib import redirect_stderr, redirect_stdout
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from threading import Event
import traceback


def verified_engine(settings, *, frozen):
    """This function also runs inside the packaged multiprocessing workers."""
    from .video import create_engine

    if bool(getattr(sys, "frozen", False)) != frozen:
        raise RuntimeError("Inference worker did not use the packaged executable")
    engine = create_engine(settings)
    if "DmlExecutionProvider" not in engine.inferenceSession.get_providers():
        raise RuntimeError("Inference worker fell back to CPU instead of DirectML")
    return engine


def _check_gui(report, root):
    import customtkinter
    import numpy as np
    from types import SimpleNamespace

    from . import gui
    from .media import image_write

    first = root / "résumé {one}.png"
    second = root / "another image.PNG"
    notes = root / "notes.txt"
    broken = root / "broken.png"
    folder = root / "folder.mp4"
    image_write(str(first), np.full((32, 48, 3), 100, np.uint8), ".png")
    image_write(str(second), np.full((32, 48, 3), 150, np.uint8), ".png")
    notes.write_text("Not input media", encoding="utf-8")
    broken.write_bytes(b"Not a valid PNG")
    folder.mkdir()

    original_mainloop = customtkinter.CTk.mainloop

    def smoke_window(window):
        try:
            window.update()
            report["gui_widgets"] = len(window.winfo_children())
            if not report["gui_widgets"]:
                raise RuntimeError("The packaged GUI has no widgets")
            report["tkdnd_version"] = window.TkdndVersion

            def drop(paths, fraction=0.25):
                # Exercise the native Windows TkDND Tcl/OLE dispatch path, not
                # just a direct Python callback. No mouse movement is needed.
                x = window.winfo_rootx() + int(window.winfo_width() * fraction)
                y = window.winfo_rooty() + 100
                payload = window.tk.call("list", *(str(path) for path in paths))
                target = window.winfo_containing(x, y) or window
                window.tk.call(
                    "tkdnd::olednd::HandleDragEnter", target._w,
                    ("CF_HDROP",), ("copy",), (), x, y, ("15",), payload,
                )
                try:
                    return window.tk.call(
                        "tkdnd::olednd::HandleDrop", target._w, (), x, y,
                        "CF_HDROP", payload,
                    )
                finally:
                    window.tk.call("tkdnd::olednd::HandleDragLeave", target._w)

            expected = [str(first.resolve()), str(second.resolve())]
            if drop([first, notes, folder, broken]) != "copy":
                raise RuntimeError("Native file drop was not accepted")
            if gui.file_widget.get_selected_file_list() != expected[:1]:
                raise RuntimeError("Drop did not filter unsupported/unreadable inputs")
            if drop([first, second]) != "copy":
                raise RuntimeError("Dropping more files into a populated list failed")
            if gui.file_widget.get_selected_file_list() != expected:
                raise RuntimeError("Repeated drops replaced or duplicated selected inputs")
            for paths, fraction in (([notes, folder], 0.25), ([broken], 0.25),
                                    ([second], 0.75)):
                if drop(paths, fraction) != "refuse_drop":
                    raise RuntimeError("Invalid/outside-input drop was accepted")
                if gui.file_widget.get_selected_file_list() != expected:
                    raise RuntimeError("Rejected drop changed the current selection")
            previous_process = getattr(gui, "process_upscale_orchestrator", None)
            gui.process_upscale_orchestrator = SimpleNamespace(is_alive=lambda: True)
            try:
                if drop([second]) != "refuse_drop":
                    raise RuntimeError("Drop changed input files during an active job")
            finally:
                gui.process_upscale_orchestrator = previous_process
            gui.clear_input_files()
            if gui.file_widget is not None or gui.selected_file_list:
                raise RuntimeError("Clear did not reset the input selection")
            if drop([second]) != "copy" or gui.file_widget.get_selected_file_list() != expected[1:]:
                raise RuntimeError("File drops stopped working after clearing the list")
            report["file_drop_verified"] = True
        finally:
            # Do not invoke on_app_close, which persists user preferences.
            window.destroy()

    customtkinter.CTk.mainloop = smoke_window
    try:
        gui.main()
    finally:
        customtkinter.CTk.mainloop = original_mainloop


def _check_bundle(root, report):
    import cv2
    import numpy as np
    import onnxruntime as ort
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import subprocess

    from .ai import AI_upscale
    from .ffmpeg import FFmpeg, UpscaleCancelled
    from .frame_store import FrameStore
    from .paths import find_by_relative_path
    from .video import VideoSettings, VideoUpscaler

    _check_gui(report, root)
    model = Path(find_by_relative_path("AI-onnx/RealESR_Gx4_fp16.onnx"))
    bundled_ffmpeg = Path(find_by_relative_path("Assets/ffmpeg.exe"))
    ffmpeg = FFmpeg()
    if not model.is_file() or not bundled_ffmpeg.is_file():
        raise RuntimeError("The default model or bundled FFmpeg is missing")
    if Path(ffmpeg.executable).resolve() != bundled_ffmpeg.resolve():
        raise RuntimeError("Self-test must use bundled FFmpeg, not a PATH installation")
    report.update({
        "model": model.name,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "bundled_models": sorted(path.name for path in model.parent.glob("*_fp16.onnx")),
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "bundled_ffmpeg": str(bundled_ffmpeg),
    })

    original_session = ort.InferenceSession

    def profiled_session(*args, **kwargs):
        options = kwargs["sess_options"]
        options.enable_profiling = True
        options.profile_file_prefix = str(root / "directml-profile")
        return original_session(*args, **kwargs)

    # Profile the production loader without changing its provider or DML options.
    # This diagnostic runs before any worker is started; restore the constructor
    # before spawning so workers use the unmodified production inference path.
    ort.InferenceSession = profiled_session
    try:
        engine = AI_upscale("RealESR_Gx4", "Auto", 1.0, 1.0, 512)
    finally:
        ort.InferenceSession = original_session
    try:
        image = np.random.default_rng(42).integers(0, 256, (32, 48, 3), dtype=np.uint8)
        result = engine.AI_orchestration(image)
    finally:
        profile = Path(engine.inferenceSession.end_profiling())
    del engine
    if result.shape != (128, 192, 3):
        raise RuntimeError(f"Incorrect model output dimensions: {result.shape}")
    executed = sorted({
        event["args"]["provider"] for event in json.loads(profile.read_text(encoding="utf-8"))
        if event.get("args", {}).get("provider")
    })
    if "DmlExecutionProvider" not in executed:
        raise RuntimeError(f"No DirectML kernels executed: {executed}")
    report["executed_providers"] = executed

    source = root / "input.mp4"
    output = root / "input_upscaled.mp4"
    cache = root / "input.mp4.qualityscaler" / "frames.sqlite3"
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.run([
        ffmpeg.executable, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=48x32:rate=4:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
    ], check=True, creationflags=flags)
    settings = VideoSettings("RealESR_Gx4", workers=2, tiles_resolution=512, keep_frames=True)
    factory = partial(verified_engine, frozen=report["frozen"])
    stop = Event()

    def pause_after_saving(message):
        print(message, flush=True)
        if message.startswith("Upscaling video"):
            stop.set()

    try:
        VideoUpscaler(
            ffmpeg=ffmpeg, engine_factory=factory, stop_event=stop,
            status=pause_after_saving,
        ).run(source, output, cache, settings)
    except UpscaleCancelled:
        pass
    else:
        raise RuntimeError("The job did not pause")
    with FrameStore(cache, read_only=True) as store:
        total = store.get_metadata("total_frames")
        saved = store.frame_count("upscaled")
        if total != 8 or not 0 < saved < total or store.get_metadata("stage") != "paused":
            raise RuntimeError("Interrupted job state was not saved correctly")
        previous_frames = {index: store.get_frame("upscaled", index)
                           for index in store.iter_indices("upscaled")}

    submitted = []

    class CountingExecutor(ProcessPoolExecutor):
        def submit(self, *args, **kwargs):
            submitted.append(True)
            return super().submit(*args, **kwargs)

    VideoUpscaler(
        ffmpeg=ffmpeg, engine_factory=factory, status=print,
        executor_factory=partial(CountingExecutor, mp_context=get_context("spawn")),
    ).run(source, output, cache, settings)
    if len(submitted) != total - saved:
        raise RuntimeError("Resume did not process exactly the missing frames")
    with FrameStore(cache, read_only=True) as store:
        if store.get_metadata("stage") != "complete" or store.frame_count("upscaled") != total:
            raise RuntimeError("Resumed job did not complete")
        for index, data in previous_frames.items():
            if store.get_frame("upscaled", index) != data:
                raise RuntimeError("Resume overwrote an already-saved frame")
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame.shape != (128, 192, 3):
                raise RuntimeError(f"Incorrect encoded frame dimensions: {frame.shape}")
            decoded += 1
    finally:
        capture.release()
    if decoded != total or fps != 4:
        raise RuntimeError(f"Incorrect output frame count/rate: {decoded} frames, {fps} fps")
    subprocess.run([
        ffmpeg.executable, "-v", "error", "-i", str(output),
        "-map", "0:a:0", "-f", "null", "-",
    ], check=True, creationflags=flags)
    report.update({
        "frames": decoded, "fps": fps, "width": 192, "height": 128,
        "frames_before_pause": saved, "frames_resumed": len(submitted),
        "audio_preserved": True, "workers": settings.workers,
        "cache_files": sorted(path.name for path in cache.parent.iterdir()),
    })


def run_self_test(report_path):
    """Write a JSON result and sibling log; return a process exit code."""
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "frozen": bool(getattr(sys, "frozen", False)),
              "python": sys.version, "platform": platform.platform()}
    log_path = report_path.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log, \
            redirect_stdout(log), redirect_stderr(log):
        try:
            with TemporaryDirectory(prefix="qualityscaler-self-test-", dir=report_path.parent) as root:
                _check_bundle(Path(root), report)
            report["status"] = "passed"
        except Exception:
            report["status"] = "failed"
            report["error"] = traceback.format_exc()
            print(report["error"], file=log)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["status"] == "passed" else 1
