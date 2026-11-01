"""Lazy, process-owned access to PyInstaller's native bootloader splash.

Only the desktop launcher opts in, after multiprocessing.freeze_support().
Importing this module or calling the public helpers elsewhere is harmless:
no Tk root, no source-install dependency on pyi_splash, and no worker IPC.
"""

from multiprocessing import current_process
import os
import sys


_owner_pid = None
_splash = None
_closed = False


def _enable_splash():
    """Opt in from the frozen desktop launcher, never a CLI or spawn worker."""
    global _owner_pid
    if (
        _owner_pid is None
        and getattr(sys, "frozen", False)
        and current_process().name == "MainProcess"
        and not any(arg in {"--self-test", "--multiprocessing-fork", "-c"}
                    for arg in sys.argv[1:])
    ):
        _owner_pid = os.getpid()


def _get_splash():
    global _splash, _closed
    # PID ownership also protects a forked child inheriting an already-open module.
    if _closed or _owner_pid != os.getpid():
        return None
    if _splash is None:
        # 6.16's bootloader sets this to 0 in subprocesses or when suppressed.
        # pyi_splash consumes the variable on import, so check only before loading.
        try:
            port = int(os.environ.get("_PYI_SPLASH_IPC", "0"))
        except ValueError:
            port = 0
        if not 0 < port < 65536:
            _closed = True
            return None
        try:
            import pyi_splash
        except (ImportError, OSError, RuntimeError):
            _closed = True
            return None
        _splash = pyi_splash
    return _splash


def update_splash(text: str, *, progress: int | None = None) -> None:
    """Send status and optional phase progress (0–100) to the owned native splash."""
    splash = _get_splash()
    if splash is None:
        return
    try:
        if splash.is_alive():
            # Plain text remains supported; the native Tcl extension consumes
            # this prefix before displaying the status. No second IPC channel.
            if progress is not None:
                text = f"QS_PROGRESS:{max(0, min(100, progress))}:{text}"
            splash.update_text(text)
    except (OSError, RuntimeError):
        # Optional feedback must never break startup or mask the real error.
        close_splash()


def close_splash() -> None:
    """Close once at desktop readiness or on startup failure; never reconnect."""
    global _closed
    splash = _get_splash()
    if _owner_pid != os.getpid() or _closed:
        return
    _closed = True
    if splash is not None:
        try:
            splash.close()
        except (OSError, RuntimeError):
            pass
