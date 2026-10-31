"""Picklable, GUI-free helper for testing the desktop spawn bootstrap."""

import os
import sys


def process_imports():
    return {
        "pid": os.getpid(),
        "gui_modules": [name for name in (
            "qualityscaler.gui", "customtkinter", "tkinter", "onnxruntime",
        ) if name in sys.modules],
    }
