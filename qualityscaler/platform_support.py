"""Best-effort Windows integration, safe to import from GUI-free workers."""

import os
import sys

from .paths import find_by_relative_path


_APP_NAME = "QualityScaler"
LOGO_PNG_PATH = find_by_relative_path(os.path.join("Assets", "logo.png"))


def prevent_sleep() -> None:
    """Keep a Windows batch awake without forcing the display to stay on."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
    except Exception as error:
        print(f"[{_APP_NAME}] Warning: could not prevent system sleep: {error}")


def allow_sleep() -> None:
    """Restore Windows sleep behavior after the batch."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception as error:
        print(f"[{_APP_NAME}] Warning: could not restore system sleep state: {error}")


def show_completion_notification() -> None:
    """Show a Windows toast if the optional notification provider is available."""
    if sys.platform != "win32":
        return
    try:
        from winotify import Notification

        Notification(
            app_id=_APP_NAME,
            title=_APP_NAME,
            msg="All files completed! :)",
            icon=LOGO_PNG_PATH,
        ).show()
    except Exception as error:
        print(f"[{_APP_NAME}] Warning: could not show system notification: {error}")
