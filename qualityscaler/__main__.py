"""Inspect and resume cached jobs without reconstructing GUI settings."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
from threading import Event

from .ffmpeg import UpscaleCancelled
from .frame_store import FrameStore
from .video import VideoSettings, VideoUpscaler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Print saved settings and progress as JSON")
    inspect.add_argument("cache", type=Path)
    resume = commands.add_parser("resume", help="Resume using settings saved in SQLite")
    resume.add_argument("cache", type=Path)
    resume.add_argument("--codec", help="Retry encoding with a different codec, e.g. x264")
    resume.add_argument("--workers", type=int)
    resume.add_argument("--gpu")
    resume.add_argument("--keep-frames", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if not args.cache.is_file():
        parser.error(f"Cache does not exist: {args.cache}")

    # Resume opens writable so SQLite can recover a hot rollback journal left
    # by a hard kill. A read-only connection cannot perform that recovery.
    with FrameStore(args.cache, read_only=args.command == "inspect") as store:
        saved = {key: store.get_metadata(key) for key in (
            "source", "settings", "output_path", "stage", "interrupted_stage",
            "total_frames", "extraction_complete", "fps", "updated_at", "error",
        )}
        saved["source_frames"] = store.frame_count("source")
        saved["upscaled_frames"] = store.frame_count("upscaled")
    if args.command == "inspect":
        print(json.dumps(saved, indent=2, ensure_ascii=False))
        return 0
    if not saved["settings"] or not saved["source"] or not saved["output_path"]:
        parser.error("Cache does not contain a complete saved job configuration")
    settings = VideoSettings(**saved["settings"])
    overrides = {key: getattr(args, key) for key in ("codec", "workers", "gpu", "keep_frames")
                 if getattr(args, key) is not None}
    settings = replace(settings, **overrides)
    stop = Event()
    previous_handler = signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    try:
        VideoUpscaler(stop_event=stop, status=print).run(
            saved["source"]["path"], saved["output_path"], args.cache, settings
        )
    except UpscaleCancelled:
        print("Paused. Committed frames are available for resume.")
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
