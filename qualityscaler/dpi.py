"""Early Windows DPI opt-in; CustomTkinter remains the scaling source of truth.

Importing this module has no Windows or GUI side effects, including in spawned
workers. Only the desktop calls configure_high_dpi(), before creating any Tk
windows. macOS keeps Tk's native Retina support; other platforms keep CTk's
default behavior.
"""

import os
import sys


_awareness_pid = None
_ctk_configured = False


def _enable_windows_dpi_awareness():
    """Opt in once per process, or leave an existing host/manifest policy alone."""
    global _awareness_pid
    if sys.platform != "win32" or _awareness_pid == os.getpid():
        return
    _awareness_pid = os.getpid()
    try:
        import ctypes

        # Match CustomTkinter's per-monitor v1 policy. Its source documents a
        # Tk window-growth bug with PMv2 and resizable(False, False), which our
        # message dialogs use. Do not change dialog behavior just to use PMv2.
        result = ctypes.windll.shcore.SetProcessDpiAwareness(2)
        if (result & 0xFFFFFFFF) in (0, 0x80070005):  # S_OK / E_ACCESSDENIED (already set)
            return
    except (ImportError, AttributeError, OSError):
        pass  # Windows before 8.1, or an unavailable optional API.
    try:
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()  # Vista/7 system-aware fallback
    except (ImportError, AttributeError, OSError):
        pass  # DPI feedback is optional, never a reason to fail startup.


def configure_high_dpi():
    """Initialize awareness before Tk, without applying a second scale factor."""
    global _ctk_configured
    if sys.platform != "win32":
        return
    _enable_windows_dpi_awareness()
    if _ctk_configured:
        return
    from customtkinter import ScalingTracker

    # CTk normally calls shcore after constructing Tk and for every toplevel.
    # Reuse our guarded, idempotent opt-in instead; do NOT deactivate automatic
    # DPI awareness, which would also disable CTk's monitor detection/scaling.
    ScalingTracker.activate_high_dpi_awareness = staticmethod(_enable_windows_dpi_awareness)
    detect_monitor_dpi = ScalingTracker.get_window_dpi_scaling

    def monitor_dpi(window):
        try:
            return detect_monitor_dpi(window)
        except (AttributeError, OSError):
            # Older Windows may lack CTk's GetDpiForMonitor API as well. Keep the
            # library's default factor rather than inventing competing DPI math.
            return 1

    ScalingTracker.get_window_dpi_scaling = staticmethod(monitor_dpi)
    _ctk_configured = True
