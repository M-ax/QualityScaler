"""Spawn-safe desktop launcher; inference workers never import the GUI."""

from multiprocessing import freeze_support


if __name__ == "__main__":
    freeze_support()
    import sys

    if sys.argv[1:2] == ["--self-test"]:
        if len(sys.argv) != 3:
            raise SystemExit("Usage: QualityScaler.exe --self-test REPORT.json")
        from qualityscaler.self_test import run_self_test

        raise SystemExit(run_self_test(sys.argv[2]))
    else:
        from qualityscaler.startup import _enable_splash, close_splash, update_splash

        try:
            _enable_splash()
            update_splash("Loading QualityScaler...", progress=72)
            from qualityscaler.gui import main

            update_splash("Preparing desktop...", progress=82)
            main()
        finally:
            # Includes import/preparation errors, SystemExit and KeyboardInterrupt.
            # Normally the desktop closes it shortly after its first full repaint.
            close_splash()
