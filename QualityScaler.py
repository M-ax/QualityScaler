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
        from qualityscaler.gui import main

        main()
