"""The 2026.4 desktop UI; processing and persistence live in backend modules.

Importing this module creates neither Tk nor multiprocessing resources. The
minimal launcher imports it only for a desktop launch, never in spawn workers.
"""

from dataclasses import dataclass, field
from json import load as json_load, dumps as json_dumps
from multiprocessing import Manager as multiprocessing_Manager, Process
from os import sep as os_separator
from os.path import (
    basename as os_path_basename,
    dirname as os_path_dirname,
    exists as os_path_exists,
    expanduser as os_path_expanduser,
    join as os_path_join,
)
from pathlib import Path
from queue import Empty
from shutil import disk_usage as shutil_disk_usage
import sys
from threading import Thread
from time import monotonic
from tkinter import BooleanVar, DISABLED, StringVar, TclError
from typing import Any, Callable, Optional
from webbrowser import open as open_browser

from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    COLOR_BGRA2RGBA,
    COLOR_GRAY2RGB,
    error as opencv_error,
    VideoCapture as opencv_VideoCapture,
    cvtColor as opencv_cvtColor,
    resize as opencv_resize,
)
from PIL.Image import open as pillow_image_open, fromarray as pillow_image_fromarray
from psutil import Process as psutil_Process, NoSuchProcess, AccessDenied
from customtkinter import (
    CTk, CTkFrame, CTkButton, CTkCheckBox, CTkEntry, CTkFont, CTkImage, CTkLabel,
    CTkOptionMenu, CTkProgressBar, CTkScrollableFrame, CTkToplevel,
    filedialog, set_appearance_mode, set_default_color_theme,
    set_widget_scaling, set_window_scaling,
)

from .ai import get_model_upscale_factor
from .catalog import (
    APP_NAME as app_name, VERSION as version, AI_models_list,
    MODEL_NAMES, VRAM_model_usage,
)
from .config import ProcessingConfig
from .drag_drop import FileDropTarget, FileDropWindow
from .encoders import EncoderOptions, encoder_capabilities, normalize_encoder_options
from .formats import VIDEO_EXTENSIONS
from .gpu import GPU, get_AI_engine_info
from .inputs import merge_input_files, supported_input_files
from .jobs import upscale_orchestrator
from .media import get_image_resolution, image_read
from .paths import OUTPUT_PATH_CODED, find_by_relative_path, prepare_output_video_directory_name
from .platform_support import prevent_sleep, allow_sleep, show_completion_notification
from .preview_window import UpscalePreview
from .startup import close_splash, update_splash
from .status import CacheSettingsConflict, VideoProgress
from .video import VideoSettings, get_resume_progress
from .widgets import StatusBar

githubme = "https://github.com/Djdefrag/QualityScaler/releases"
telegramme = "https://linktr.ee/j3ngystudio"
app_name_color = "#F274EE"
background_color = "#000000"
widget_background_color = "#181818"
text_color = "#C8C8C8"
CARD_BACKGROUND_COLOR = "#1A1A1A"
CARD_BORDER_COLOR = "#2B2B2B"
CARD_TITLE_COLOR = "#ECECEC"
CARD_VALUE_COLOR = "#C8C8C8"
CARD_MUTED_COLOR = "#7E7E7E"
CARD_FAINT_COLOR = "#6A6A6A"
CARD_ACCENT_COLOR = "#5AA9FF"
RESUME_BADGE_COLOR = "#16261C"
RESUME_ACCENT_COLOR = "#4ADE80"
RESUME_BAR_DIM_COLOR = "#1E4A33"
MESSAGE_SUCCESS_COLOR = "#4ADE80"
MESSAGE_WARNING_COLOR = "#F0B85A"
MESSAGE_ERROR_COLOR = "#FF5C5C"
UI_ACCENT_COLOR = CARD_ACCENT_COLOR
UI_BORDER_COLOR = CARD_BORDER_COLOR
UI_CORNER_RADIUS = 6

zoom_option_list = ["50%", "75%", "100%", "125%", "150%", "175%"]
AI_multithreading_list = ["OFF", "2 threads", "4 threads", "6 threads", "8 threads"]
sharpening_list = ["OFF", "Low", "High"]
gpus_list = ["No GPU found"]
keep_frames_list = ["OFF", "ON"]
image_extension_list = [".png", ".jpg", ".bmp", ".tiff"]
video_extension_list = [".mp4", ".mkv", ".avi", ".mov"]
video_codec_list = [
    "x264", "x265", "----",
    "h264_nvenc", "hevc_nvenc", "----",
    "h264_amf", "hevc_amf", "----",
    "h264_qsv", "hevc_qsv",
]
DOCUMENT_PATH = os_path_join(os_path_expanduser("~"), "Documents")
USER_PREFERENCE_PATH = os_path_join(
    DOCUMENT_PATH, f"{app_name}_{version}_userpreference.json"
)
MIN_FREE_DISK_SPACE_GB = 2
COMPLETED_STATUS = "Completed"
ERROR_STATUS = "Error"
STOP_STATUS = "Stop"
CLOSE_APP_STATUS = "CloseApp"


@dataclass
class UserPreferences:
    app_zoom: str = "100%"
    ai_model: str = AI_models_list[0]
    ai_multithreading: str = AI_multithreading_list[0]
    gpu: str = gpus_list[0]
    keep_frames: bool = True
    image_extension: str = image_extension_list[0]
    video_extension: str = video_extension_list[0]
    video_codec: str = video_codec_list[0]
    video_preset: str = "Default"
    video_rate_control: str = "Bitrate"
    video_crf: str = "23"
    video_bitrate: str = "50000"
    sharpening: str = sharpening_list[1]
    output_path: str = OUTPUT_PATH_CODED
    input_resize_factor: str = "50"
    output_resize_factor: str = "100"
    vram_limiter: str = "4"


@dataclass
class AppState:
    preferences: UserPreferences
    window: Optional[CTk] = None
    input_panel: Optional[Any] = None
    settings_panel: Optional[Any] = None
    settings_content: Optional[Any] = None
    header: Optional[Any] = None
    action_panel: Optional[Any] = None
    status_bar: Optional[StatusBar] = None
    upscale_preview: Optional[UpscalePreview] = None
    option_rows: dict = field(default_factory=dict)
    info_message: Optional[StringVar] = None
    selected_output_path: Optional[StringVar] = None
    selected_input_resize_factor: Optional[StringVar] = None
    selected_output_resize_factor: Optional[StringVar] = None
    selected_VRAM_limiter: Optional[StringVar] = None
    selected_video_codec: Optional[StringVar] = None
    selected_video_crf: Optional[StringVar] = None
    selected_video_bitrate: Optional[StringVar] = None
    file_widget: Optional["FileWidget"] = None
    process_upscale_orchestrator: Optional[Any] = None
    processing_config: Optional[ProcessingConfig] = None
    process_status_q: Optional[Any] = None
    event_stop_upscale_process: Optional[Any] = None
    selected_file_list: list[str] = field(default_factory=list)
    input_widgets: tuple = ()
    action_button: Optional[Any] = None
    processing: bool = False
    stopping: bool = False
    closing: bool = False
    pending_upscale_status: Optional[str | CacheSettingsConflict] = None
    stop_deadline: Optional[float] = None
    poll_after_id: Optional[str] = None
    preview_after_id: Optional[str] = None
    after_ids: set = field(default_factory=set)
    traces: list = field(default_factory=list)


app_state: Optional[AppState] = None
ROW_AI_MODEL = 0
ROW_AI_SHARPENING = 1
ROW_AI_MULTITHREADING = 2
ROW_RESOLUTION = 3
ROW_GPU = 4
ROW_OUTPUT_FORMAT = 5
ROW_CODEC = 6
ROW_ENCODER_PRESET = 7
ROW_ENCODER_QUALITY = 8
ROW_OUTPUT_PATH = 9
little_textbox_width = 74
little_menu_width = 98


def check_if_file_is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def lerp_hex(color_a: str, color_b: str, t: float) -> str:
    a = (int(color_a[1:3], 16), int(color_a[3:5], 16), int(color_a[5:7], 16))
    b = (int(color_b[1:3], 16), int(color_b[3:5], 16), int(color_b[5:7], 16))
    r, g, blue = (round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return f"#{r:02X}{g:02X}{blue:02X}"


def _schedule(delay: int, callback: Callable, *, during_close: bool = False) -> Optional[str]:
    """Schedule on Tk's thread and retain handles for deterministic shutdown."""
    state = app_state
    if state is None or (state.closing and not during_close):
        return None

    def run():
        state.after_ids.discard(handle)
        if not state.closing or during_close:
            callback()

    handle = state.window.after(delay, run)
    state.after_ids.add(handle)
    return handle


def _cancel_after(handle):
    if handle is None or app_state.window is None:
        return
    try:
        app_state.window.after_cancel(handle)
    except TclError:
        pass
    app_state.after_ids.discard(handle)


class MessageBox(CTkToplevel):
    def __init__(self, messageType, title, subtitle, default_value, option_list,
                 *, action_text=None, action_command=None, action_option_text=None):
        super().__init__(master=App.get_app_window())
        self.configure(fg_color=background_color)
        self._messageType = messageType
        self._title = title
        self._subtitle = subtitle
        self._default_value = default_value
        self._option_list = option_list
        self._action_text = action_text
        self._action_command = action_command
        self._action_option_text = action_option_text
        self._action_option = BooleanVar(master=self, value=False) if action_option_text else None
        self.title("")
        self.lift()
        self.attributes("-topmost", True)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._create_after = self.after(10, self._create_widgets)
        self.resizable(False, False)
        self.grab_set()

    def destroy(self):
        handle = getattr(self, "_create_after", None)
        if handle is not None:
            self.after_cancel(handle)
            self._create_after = None
        super().destroy()

    def _ok_event(self, event=None):
        self._action_command = None
        self.grab_release()
        self.destroy()

    def _on_closing(self):
        self._ok_event()

    def _action_event(self):
        command = self._action_command
        if command is None:
            return
        self._action_command = None  # A second click cannot start another retry.
        option = self.__dict__.get("_action_option")
        value = option.get() if option is not None else None
        self._ok_event()
        if option is None:
            command()
        else:
            command(value)

    def _copy_log_to_clipboard(self, copy_button):
        self.clipboard_clear()
        self.clipboard_append("\n".join(self._option_list))
        self.update()
        copy_button.configure(text="Copied!")

    def _create_widgets(self):
        self._create_after = None
        self.grid_columnconfigure(0, weight=1)
        tint = UI_ACCENT_COLOR if self._messageType == "info" else MESSAGE_ERROR_COLOR
        CTkLabel(self, text=self._title, font=bold22, text_color=tint,
                 anchor="w", justify="left").grid(
            row=0, column=0, padx=25, pady=(20, 0), sticky="ew"
        )
        if self._default_value is not None:
            CTkLabel(self, text=f"Default: {self._default_value}", font=bold17,
                     text_color=CARD_MUTED_COLOR, anchor="w").grid(
                row=1, column=0, padx=25, sticky="ew"
            )
        CTkLabel(self, text=self._subtitle, font=bold14, text_color=CARD_VALUE_COLOR,
                 anchor="w", justify="left", wraplength=600).grid(
            row=2, column=0, padx=25, pady=(0, 17), sticky="ew"
        )
        for index, option in enumerate(self._option_list, start=3):
            CTkLabel(self, text=option, width=600, height=45, font=bold13,
                     anchor="w", justify="left", wraplength=600, text_color=CARD_VALUE_COLOR,
                     fg_color=CARD_BACKGROUND_COLOR, corner_radius=10).grid(
                row=index, column=0, padx=25, pady=4, sticky="ew"
            )
        button_row = 3 + len(self._option_list)
        if self._action_option is not None:
            CTkCheckBox(
                self, text=self._action_option_text, variable=self._action_option,
                font=bold13, text_color=CARD_VALUE_COLOR,
            ).grid(row=button_row, column=0, padx=25, pady=(12, 0), sticky="w")
            button_row += 1
        buttons = CTkFrame(self, fg_color="transparent")
        buttons.grid(row=button_row, column=0,
                     padx=20, pady=20, sticky="e")
        style = dict(width=125, font=bold11, border_width=2,
                     corner_radius=UI_CORNER_RADIUS, fg_color=CARD_BACKGROUND_COLOR,
                     hover_color=widget_background_color, text_color="#E0E0E0",
                     border_color=UI_ACCENT_COLOR)
        if self._messageType == "error":
            copy_button = CTkButton(buttons, text="Copy log", **style)
            copy_button.configure(command=lambda: self._copy_log_to_clipboard(copy_button))
            copy_button.pack(side="left", padx=(0, 10))
        if self._action_command is not None:
            CTkButton(
                buttons, text=self._action_text, command=self._action_event,
                **{**style, "width": 240},
            ).pack(side="left", padx=(0, 10))
        dismiss_text = "Cancel" if self._action_command is not None else "OK"
        CTkButton(buttons, text=dismiss_text, command=self._ok_event, **style).pack(side="left")


class FileWidget(CTkScrollableFrame):
    """Upstream file cards with lifecycle-bound thumbnails and animations."""

    def __init__(self, master, selected_file_list, upscale_factor=1,
                 input_resize_factor=0, output_resize_factor=0, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.file_list = list(selected_file_list)
        self.unreadable_files = []
        self.upscale_factor = upscale_factor
        self.input_resize_factor = input_resize_factor
        self.output_resize_factor = output_resize_factor
        self.index_row = 1
        self.ui_components = []
        self._animation_ids = {}
        self.add_clean_button()
        self._render_cards()

    def _cancel_animations(self):
        for handle, owner in self._animation_ids.items():
            try:
                owner.after_cancel(handle)
            except TclError:
                pass
        self._animation_ids.clear()

    def _animate_after(self, owner, delay, callback):
        def run():
            self._animation_ids.pop(handle, None)
            if owner.winfo_exists():
                callback()
        handle = owner.after(delay, run)
        self._animation_ids[handle] = owner

    def destroy(self):
        self._cancel_animations()
        # A global functools.cache on a bound method retains the entire
        # destroyed scroll frame and all its Tk images indefinitely.
        self.ui_components.clear()
        super().destroy()

    def _destroy_(self):
        return clear_input_files()

    def _render_cards(self):
        readable = []
        for file_path in self.file_list:
            try:
                item = self._create_file_card(file_path)
            except (OSError, ValueError, opencv_error):
                self.unreadable_files.append(file_path)
                continue
            readable.append(file_path)
            self.ui_components.append(item)
        self.file_list = readable

    def _remove_file(self, file_path):
        if inputs_are_busy() or file_path not in self.file_list:
            return
        close_upscale_preview()
        self.file_list.remove(file_path)
        if not self.file_list:
            self._destroy_()
            return
        self.clean_file_list()
        self._render_cards()
        app_state.selected_file_list = list(self.file_list)
        _display_status(f"Ready — {len(self.file_list)} input file(s)")

    def _create_file_card(self, file_path):
        is_video, width, height, num_frames, frame_rate = self._read_media_properties(file_path)
        file_icon = self.extract_file_icon(file_path)
        card = CTkFrame(self, fg_color=CARD_BACKGROUND_COLOR, corner_radius=12,
                        border_width=2, border_color=CARD_BORDER_COLOR)
        card.grid(row=self.index_row, column=0, columnspan=3, padx=6,
                  pady=(3, 9), sticky="ew")
        card.grid_columnconfigure(1, weight=1)
        thumbnail = CTkLabel(card, text="", image=file_icon)
        thumbnail.grid(row=0, column=0, padx=(12, 14), pady=12, sticky="n")
        remove_button = CTkButton(
            card, command=lambda: self._remove_file(file_path), text="X",
            width=24, height=24, font=bold11, fg_color="transparent",
            hover_color="#8C3B3B", border_width=2, border_color=MESSAGE_ERROR_COLOR,
            text_color=MESSAGE_ERROR_COLOR, corner_radius=UI_CORNER_RADIUS,
        )
        remove_button.grid(row=0, column=2, padx=(0, 8), pady=12, sticky="n")
        for widget in (remove_button, remove_button._text_label):
            widget.bind("<Enter>", lambda e: remove_button._text_label.configure(fg="#FFFFFF"), add="+")
            widget.bind("<Leave>", lambda e: remove_button._text_label.configure(fg=MESSAGE_ERROR_COLOR), add="+")
        content = CTkFrame(card, fg_color="transparent")
        content.grid(row=0, column=1, sticky="ew", padx=(0, 14), pady=11)
        content.grid_columnconfigure(0, weight=1)
        CTkLabel(content, text=os_path_basename(file_path), font=bold14,
                 text_color=CARD_TITLE_COLOR, anchor="w", justify="left").grid(
            row=0, column=0, sticky="ew"
        )
        CTkLabel(
            content,
            text=self._format_source_meta(is_video, width, height, num_frames, frame_rate),
            font=bold12, text_color=CARD_MUTED_COLOR, anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(2, 0))
        dynamic_section = self._create_dynamic_section(
            content, 2, is_video, file_path, width, height
        )
        self.index_row += 1
        return dict(card=card, content=content, dynamic_row=2, is_video=is_video,
                    file_path=file_path, width=width, height=height,
                    dynamic_section=dynamic_section, icon=file_icon)

    def _create_dynamic_section(self, content, start_row, is_video, file_path, width, height):
        resume_percent = get_video_resume_progress(file_path) if is_video else None
        has_pipeline = all((self.input_resize_factor, self.output_resize_factor, self.upscale_factor))
        display_percent = resume_percent if resume_percent is not None else 0
        if not is_video and not has_pipeline:
            return None
        section = CTkFrame(content, fg_color="transparent")
        section.grid_columnconfigure(0, weight=1)
        section.grid(row=start_row, column=0, sticky="ew")
        inner_row = 0
        if is_video:
            badge = self._create_resume_badge(section, display_percent)
            badge.grid(row=inner_row, column=0, sticky="ew", pady=(8, 0))
            inner_row += 1
        if has_pipeline:
            CTkFrame(section, fg_color=CARD_BORDER_COLOR, height=1).grid(
                row=inner_row, column=0, sticky="ew", pady=(7, 6)
            )
            pipeline_table = self._create_pipeline_table(
                section, self._compute_pipeline_rows(width, height)
            )
            pipeline_table.grid(row=inner_row + 1, column=0, sticky="ew")
        return section

    def add_clean_button(self):
        style = dict(height=28, font=bold11, border_width=2,
                     corner_radius=UI_CORNER_RADIUS, fg_color="#282828",
                     text_color="#E0E0E0", border_color=UI_ACCENT_COLOR)
        CTkButton(self, command=open_files_action, text="ADD FILES", width=110,
                  **style).grid(row=0, column=0, pady=7, padx=7, sticky="w")
        CTkButton(self, command=clear_input_files, text="CLEAN", image=clear_icon,
                  width=90, **style).grid(row=0, column=2, pady=7, padx=(0, 7))

    def extract_file_icon(self, file_path):
        if check_if_file_is_video(file_path):
            video_cap = opencv_VideoCapture(file_path)
            try:
                ret, source_icon = video_cap.read()
            finally:
                video_cap.release()
            if not ret or source_icon is None:
                raise ValueError("Unreadable video preview")
        else:
            source_icon = image_read(file_path)
        if source_icon is None:
            raise ValueError("Unreadable image preview")
        if source_icon.ndim == 2:
            source_icon = opencv_cvtColor(source_icon, COLOR_GRAY2RGB)
        elif source_icon.shape[2] == 4:
            source_icon = opencv_cvtColor(source_icon, COLOR_BGRA2RGBA)
        else:
            source_icon = opencv_cvtColor(source_icon, COLOR_BGR2RGB)
        ratio = min(60 / source_icon.shape[0], 60 / source_icon.shape[1])
        size = (max(1, int(source_icon.shape[1] * ratio)),
                max(1, int(source_icon.shape[0] * ratio)))
        source_icon = opencv_resize(source_icon, size)
        return CTkImage(pillow_image_fromarray(source_icon), size=size)

    def _read_media_properties(self, file_path):
        if check_if_file_is_video(file_path):
            cap = opencv_VideoCapture(file_path)
            try:
                width = round(cap.get(CAP_PROP_FRAME_WIDTH))
                height = round(cap.get(CAP_PROP_FRAME_HEIGHT))
                num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
                fps = cap.get(CAP_PROP_FPS)
            finally:
                cap.release()
            if width <= 0 or height <= 0 or num_frames <= 0:
                raise ValueError("Unreadable video")
            from math import isfinite
            frame_rate = fps if isfinite(fps) and fps > 0 else 30.0
            return True, width, height, num_frames, frame_rate
        image = image_read(file_path)
        if image is None:
            raise ValueError("Unreadable image")
        height, width = get_image_resolution(image)
        return False, width, height, 0, 0.0

    def _format_source_meta(self, is_video, width, height, num_frames, frame_rate):
        if not is_video:
            return f"{width}×{height}"
        duration = num_frames / frame_rate if frame_rate else 0
        minutes = int(duration / 60)
        seconds = round(duration % 60)
        return f"{minutes}m {seconds}s  |  {width}×{height}  |  {round(frame_rate, 1)} fps"

    def _compute_pipeline_rows(self, width, height):
        input_width = int(width * (self.input_resize_factor / 100))
        input_height = int(height * (self.input_resize_factor / 100))
        ai_width = int(input_width * self.upscale_factor)
        ai_height = int(input_height * self.upscale_factor)
        output_width = int(ai_width * (self.output_resize_factor / 100))
        output_height = int(ai_height * (self.output_resize_factor / 100))
        return [
            ("Input", f"{self.input_resize_factor}%", f"{input_width}×{input_height}", False),
            ("AI", f"x{self.upscale_factor}", f"{ai_width}×{ai_height}", True),
            ("Output", f"{self.output_resize_factor}%", f"{output_width}×{output_height}", False),
        ]

    def _create_resume_badge(self, parent, percent):
        badge = CTkFrame(parent, fg_color=RESUME_BADGE_COLOR, corner_radius=8)
        badge.grid_columnconfigure(1, weight=1)
        CTkLabel(badge, text="Completed", font=bold12, text_color=RESUME_ACCENT_COLOR,
                 anchor="w").grid(row=0, column=0, padx=(10, 10), pady=7, sticky="w")
        bar = CTkProgressBar(badge, height=8, corner_radius=4,
                             fg_color=CARD_BORDER_COLOR, progress_color=RESUME_ACCENT_COLOR)
        bar.set(0)
        bar.grid(row=0, column=1, pady=7, sticky="ew")
        percent_label = CTkLabel(badge, text="0%", font=bold12,
                                 text_color=RESUME_ACCENT_COLOR, width=42, anchor="e")
        percent_label.grid(row=0, column=2, padx=(10, 10), pady=7, sticky="e")
        self._animate_progress_bar(bar, percent / 100, percent_label, percent)
        return badge

    def _animate_progress_bar(self, bar, target, percent_label=None, percent=0, current=0.0):
        if not bar.winfo_exists():
            return
        current += (target - current) * 0.08
        ratio = current / target if target > 0 else 1.0
        bar.configure(progress_color=lerp_hex(RESUME_BAR_DIM_COLOR, RESUME_ACCENT_COLOR, min(1.0, ratio)))
        if percent_label is not None and percent_label.winfo_exists():
            percent_label.configure(text=f"{round(current * 100)}%")
        if target - current < 0.005:
            bar.set(target)
            if percent_label is not None and percent_label.winfo_exists():
                percent_label.configure(text=f"{percent}%")
            if percent >= 100:
                self._glow_bar(bar)
            else:
                bar.configure(progress_color=RESUME_ACCENT_COLOR)
            return
        bar.set(current)
        self._animate_after(bar, 20, lambda: self._animate_progress_bar(
            bar, target, percent_label, percent, current
        ))

    def _glow_bar(self, bar, step=0, sequence=None):
        if not bar.winfo_exists():
            return
        if sequence is None:
            peak = "#C4FFDC"
            up = [lerp_hex(RESUME_ACCENT_COLOR, peak, i / 6) for i in range(7)]
            down = [lerp_hex(peak, RESUME_ACCENT_COLOR, i / 6) for i in range(1, 7)]
            sequence = up + down + up + down
        if step >= len(sequence):
            bar.configure(progress_color=RESUME_ACCENT_COLOR)
            return
        bar.configure(progress_color=sequence[step])
        self._animate_after(bar, 45, lambda: self._glow_bar(bar, step + 1, sequence))

    def _create_pipeline_table(self, parent, pipeline_rows):
        table = CTkFrame(parent, fg_color="transparent")
        table.grid_columnconfigure(0, minsize=96)
        table.grid_columnconfigure(1, minsize=96)
        table.grid_columnconfigure(2, weight=1)
        for row_index, (label, detail, resolution, is_ai) in enumerate(pipeline_rows):
            colors = (CARD_ACCENT_COLOR,) * 3 if is_ai else (
                CARD_MUTED_COLOR, CARD_FAINT_COLOR, CARD_VALUE_COLOR
            )
            for column, (text, color) in enumerate(zip((label, detail, resolution), colors)):
                CTkLabel(table, text=text, font=bold12, text_color=color,
                         anchor="w", height=20).grid(
                    row=row_index, column=column, sticky="w", pady=0,
                    padx=(0, 14) if column == 2 else 0,
                )
        return table

    def clean_file_list(self):
        self._cancel_animations()
        self.index_row = 1
        for item in self.ui_components:
            item["card"].destroy()
        self.ui_components = []

    def refresh_pipeline(self):
        self._cancel_animations()
        for item in self.ui_components:
            if item["dynamic_section"] is not None:
                item["dynamic_section"].destroy()
            item["dynamic_section"] = self._create_dynamic_section(
                item["content"], item["dynamic_row"], item["is_video"],
                item["file_path"], item["width"], item["height"],
            )

    def get_selected_file_list(self):
        return self.file_list

    def highlight_active_file(self, file_number):
        for index, item in enumerate(self.ui_components):
            if item["card"].winfo_exists():
                item["card"].configure(
                    border_color=CARD_ACCENT_COLOR if index == file_number - 1 else CARD_BORDER_COLOR
                )

    def clear_active_highlight(self):
        for item in self.ui_components:
            if item["card"].winfo_exists():
                item["card"].configure(border_color=CARD_BORDER_COLOR)

    def set_upscale_factor(self, value):
        self.upscale_factor = value

    def set_input_resize_factor(self, value):
        self.input_resize_factor = value

    def set_output_resize_factor(self, value):
        self.output_resize_factor = value


# GUI state, processing contracts, and validation ----------------------------

def get_values_for_file_widget():
    factors = []
    for variable in (app_state.selected_input_resize_factor, app_state.selected_output_resize_factor):
        try:
            factors.append(int(float(str(variable.get()))))
        except (ValueError, TypeError, OverflowError):
            factors.append(0)
    return get_upscale_factor(), *factors


def update_file_widget(*_args):
    if app_state is None:
        return
    _cancel_after(app_state.preview_after_id)
    app_state.preview_after_id = None
    file_widget = app_state.file_widget
    if file_widget is None or app_state.closing:
        return
    upscale_factor, input_factor, output_factor = get_values_for_file_widget()
    file_widget.set_upscale_factor(upscale_factor)
    file_widget.set_input_resize_factor(input_factor)
    file_widget.set_output_resize_factor(output_factor)
    file_widget.refresh_pipeline()
    if app_state.upscale_preview is not None:
        app_state.upscale_preview.settings_changed()


def schedule_file_widget_update(*_args):
    """Coalesce typing into one preview rebuild, rather than one per key."""
    if app_state is None or app_state.closing or app_state.file_widget is None:
        return
    _cancel_after(app_state.preview_after_id)
    app_state.preview_after_id = _schedule(200, update_file_widget)


def get_current_ai_model():
    return app_state.preferences.ai_model


def get_current_ai_multithreading():
    value = app_state.preferences.ai_multithreading
    return 1 if value == "OFF" else int(value.split()[0])


def get_current_sharpening_amount():
    return {"OFF": 0, "Low": 0.3, "High": 0.5}.get(app_state.preferences.sharpening, 0)


def _positive_integer(variable, label):
    try:
        value = int(float(str(variable.get())))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be a number") from None
    if value <= 0:
        raise ValueError(f"{label} must be a value > 0")
    return value


def _video_encoder_options():
    preferences = app_state.preferences
    options = dict(preset=preferences.video_preset, rate_control=preferences.video_rate_control)
    if preferences.video_rate_control == "CRF":
        value = (app_state.selected_video_crf.get() if app_state.selected_video_crf is not None
                 else preferences.video_crf)
        try:
            options["crf"] = float(value)
        except (ValueError, TypeError, OverflowError):
            raise ValueError("CRF must be a number between 0 and 51") from None
    else:
        value = (app_state.selected_video_bitrate.get() if app_state.selected_video_bitrate is not None
                 else preferences.video_bitrate)
        try:
            options["bitrate_kbps"] = int(value)
        except (ValueError, TypeError, OverflowError):
            raise ValueError("Video bitrate must be a positive whole number in kb/s") from None
    result = EncoderOptions(**options)
    result.validate(preferences.video_codec)
    return result


def _normalize_encoder_preferences(preferences):
    options = normalize_encoder_options(preferences.video_codec, EncoderOptions(
        preset=preferences.video_preset, rate_control=preferences.video_rate_control,
    ))
    preferences.video_preset = options.preset
    preferences.video_rate_control = options.rate_control


def refresh_encoder_controls():
    _normalize_encoder_preferences(app_state.preferences)
    for row in (ROW_ENCODER_PRESET, ROW_ENCODER_QUALITY):
        background = app_state.option_rows.pop(row, None)
        if background is not None:
            background.destroy()
    if app_state.settings_content is not None:
        App.place_video_encoder_menus()


def _processing_config(selected_files):
    """One conversion rule for job startup and SQLite resume identity."""
    preferences = app_state.preferences
    selected_files = list(selected_files)
    if preferences.ai_model not in MODEL_NAMES:
        raise ValueError("Please select the AI model")
    if preferences.video_codec not in video_codec_list or preferences.video_codec == "----":
        raise ValueError("Please select the video codec")
    input_factor = _positive_integer(app_state.selected_input_resize_factor, "Input resolution %") / 100
    output_factor = _positive_integer(app_state.selected_output_resize_factor, "Output resolution %") / 100
    vram = _positive_integer(app_state.selected_VRAM_limiter, "GPU VRAM value")
    return ProcessingConfig(
        selected_file_list=selected_files,
        selected_output_path=app_state.selected_output_path.get(),
        selected_AI_model=preferences.ai_model,
        selected_AI_multithreading=get_current_ai_multithreading(),
        input_resize_factor=input_factor,
        output_resize_factor=output_factor,
        selected_gpu=GPU.device_id_for(preferences.gpu),
        tiles_resolution=int(VRAM_model_usage[preferences.ai_model] * vram * 100),
        selected_sharpening_amount=get_current_sharpening_amount(),
        selected_keep_frames=preferences.keep_frames,
        selected_image_extension=preferences.image_extension,
        selected_video_extension=preferences.video_extension,
        selected_video_codec=preferences.video_codec,
        selected_video_options=(
            _video_encoder_options() if any(check_if_file_is_video(path) for path in selected_files)
            else EncoderOptions()
        ),
    )


def get_video_resume_progress(video_path):
    # The backend owns cache paths, source/settings identity and readonly
    # SQLite access. Never scan per-frame folders or synthesize stale 100% flags.
    try:
        config = _processing_config((video_path,))
        settings = VideoSettings(
            model=config.selected_AI_model, gpu=config.selected_gpu,
            input_resize_factor=config.input_resize_factor,
            output_resize_factor=config.output_resize_factor,
            tiles_resolution=config.tiles_resolution,
            workers=config.selected_AI_multithreading,
            sharpening_amount=config.selected_sharpening_amount,
            codec=config.selected_video_codec, extension=config.selected_video_extension,
            keep_frames=config.selected_keep_frames,
            encoder_options=config.selected_video_options,
        )
    except (ValueError, TypeError, OverflowError):
        return None
    return get_resume_progress(video_path, config.selected_output_path, settings)


def build_processing_config():
    selected = app_state.file_widget.get_selected_file_list() if app_state.file_widget is not None else []
    if not selected:
        app_state.info_message.set("Please select a file")
        return None
    try:
        config = _processing_config(selected)
    except ValueError as error:
        app_state.info_message.set(str(error))
        return None
    warning = check_disk_space(config.selected_output_path, config.selected_file_list)
    if warning is not None:
        app_state.info_message.set("Not enough disk space")
        show_disk_space_error_message(warning)
        return None
    app_state.selected_file_list = list(selected)
    return config


def show_error_message(exception, *, clear_cache_command=None):
    if clear_cache_command is None:
        MessageBox("error", "Upscale error", "Please report the error on Github or Telegram",
                   None, [f"\n {exception} \n"])
    else:
        MessageBox(
            "error", "Upscale error",
            "The cached video frames cannot be reused with the selected settings.\n"
            "Current UI settings are used by default.",
            None, [
                f"\n {exception} \n",
                "Clear cache and continue deletes this video's cached source and AI frames, "
                "then restarts it. Other caches are untouched. If the current output folder "
                "differs from the failed run's folder, its old cache is kept. "
                "Any existing output is replaced only after the new video encodes successfully. "
                "Cancel leaves the cache unchanged; you can edit settings before trying again.",
            ],
            action_text="Clear cache and continue", action_command=clear_cache_command,
            action_option_text="Use settings from failed run",
        )


def show_disk_space_error_message(warning):
    MessageBox("error", "Not enough disk space",
               "Free up some space on the output drive and try again", None, [f"\n {warning} \n"])


def check_disk_space(output_path, selected_file_list):
    videos = [path for path in selected_file_list if check_if_file_is_video(path)]
    if not videos:
        return None
    # Inputs can span drives; when output is alongside input, check each one.
    paths = {os_path_dirname(path) for path in videos} if output_path == OUTPUT_PATH_CODED else {output_path}
    for path in paths:
        while path and not os_path_exists(path):
            parent = os_path_dirname(path)
            if parent == path:
                break
            path = parent
        if not path or not os_path_exists(path):
            continue
        try:
            free_gb = shutil_disk_usage(path).free / (1024 ** 3)
        except OSError:
            continue
        if free_gb < MIN_FREE_DISK_SPACE_GB:
            return (
                f"Not enough free disk space on the output drive ({free_gb:.1f} GB free, "
                f"at least {MIN_FREE_DISK_SPACE_GB} GB recommended)\n{path}"
            )
    return None


def get_upscale_factor():
    model = get_current_ai_model()
    return get_model_upscale_factor(model) if model in MODEL_NAMES else 0


def inputs_are_busy():
    if app_state is None:
        return False
    process = app_state.process_upscale_orchestrator
    return app_state.processing or app_state.stopping or (
        process is not None and process.is_alive()
    )


def accept_input_drop(event):
    """Only the left input pane accepts drops, and never during a job."""
    if app_state is None or app_state.closing or inputs_are_busy():
        return False
    panel = app_state.input_panel
    if panel is None:
        return False
    x = event.x_root - panel.winfo_rootx()
    y = event.y_root - panel.winfo_rooty()
    return 0 <= x < panel.winfo_width() and 0 <= y < panel.winfo_height()


def select_input_files(paths):
    """Shared picker/drop path: append readable media and preserve selection."""
    if inputs_are_busy():
        app_state.info_message.set("Stop the current job before changing input files")
        return False
    paths = list(paths)
    supported = supported_input_files(paths)
    if not supported:
        app_state.info_message.set("No supported input files (folders and other files are ignored)")
        return False
    existing = app_state.file_widget.get_selected_file_list() if app_state.file_widget is not None else []
    selected = merge_input_files(existing, supported)
    if selected == existing:
        app_state.info_message.set("These files are already selected")
        return True
    upscale_factor, input_factor, output_factor = get_values_for_file_widget()
    new_widget = FileWidget(
        master=app_state.input_panel, selected_file_list=selected, upscale_factor=upscale_factor,
        input_resize_factor=input_factor, output_resize_factor=output_factor,
        fg_color=background_color, bg_color=background_color,
    )
    readable = new_widget.get_selected_file_list()
    if not any(path in readable for path in supported):
        new_widget.destroy()
        app_state.info_message.set("Unable to read the selected images or videos")
        return False
    close_upscale_preview()
    if app_state.file_widget is not None:
        app_state.file_widget.destroy()
    for widget in app_state.input_widgets:
        widget.destroy()
    app_state.input_widgets = ()
    app_state.file_widget = new_widget
    app_state.selected_file_list = list(readable)
    new_widget.grid(row=0, column=0, rowspan=2, sticky="nsew")
    skipped = len(paths) - len(supported) + len(new_widget.unreadable_files)
    message = f"Ready — {len(readable)} input file(s)"
    if skipped:
        message += f" ({skipped} skipped)"
    _display_status(message)
    return True


def clear_input_files():
    if inputs_are_busy():
        app_state.info_message.set("Stop the current job before changing input files")
        return False
    close_upscale_preview()
    if app_state.file_widget is not None:
        app_state.file_widget.destroy()
    app_state.file_widget = None
    app_state.selected_file_list = []
    App.place_loadFile_section()
    _display_status("Drop input files here or use SELECT FILES")
    return True


def open_files_action():
    if inputs_are_busy():
        app_state.info_message.set("Stop the current job before changing input files")
        return
    paths = filedialog.askopenfilenames()
    if paths:
        select_input_files(paths)


def open_output_path_action():
    value = filedialog.askdirectory()
    app_state.preferences.output_path = value or OUTPUT_PATH_CODED
    app_state.selected_output_path.set(app_state.preferences.output_path)


def close_upscale_preview():
    preview = app_state.upscale_preview
    app_state.upscale_preview = None
    if preview is not None:
        preview.destroy()


def open_upscale_preview():
    if app_state.closing:
        return
    if inputs_are_busy():
        app_state.info_message.set("Stop the current job before opening an upscale preview")
        return
    if app_state.upscale_preview is not None:
        app_state.upscale_preview.deiconify()
        app_state.upscale_preview.lift()
        app_state.upscale_preview.focus_set()
        return
    selected = app_state.file_widget.get_selected_file_list() if app_state.file_widget else []
    if not selected:
        app_state.info_message.set("Please select a file to preview")
        return
    try:
        _processing_config(selected)  # Preview does not need output-drive disk space.
        app_state.upscale_preview = UpscalePreview(
            app_state.window, selected, lambda: _processing_config(selected),
            lambda: setattr(app_state, "upscale_preview", None),
        )
    except (ValueError, OSError, RuntimeError, TclError) as error:
        app_state.info_message.set(f"Cannot open preview: {error}")


def apply_app_zoom(zoom):
    set_window_scaling(zoom)
    set_widget_scaling(zoom)


def apply_auto_vram_for_gpu(selected_gpu, fallback=None):
    detected = GPU.vram_for(selected_gpu)
    value = str(detected) if detected is not None else fallback
    if value is not None:
        app_state.selected_VRAM_limiter.set(value)
        app_state.preferences.vram_limiter = value


def apply_auto_codec_for_gpu(selected_gpu):
    codec = GPU.codec_for(selected_gpu)
    if codec is not None and codec in video_codec_list:
        app_state.preferences.video_codec = codec
        if app_state.selected_video_codec is not None:
            app_state.selected_video_codec.set(codec)
        refresh_encoder_controls()


def open_info_messagebox(title, subtitle, option_list):
    MessageBox("info", title, subtitle, None, option_list)


# Process control: Tk owns UI updates; no GUI callbacks run on worker threads.

def _display_status(status: str | VideoProgress):
    """Keep the general message and both video rows in sync on Tk's thread."""
    progress = status if isinstance(status, VideoProgress) else None
    message = progress.message if progress is not None else status
    app_state.info_message.set(message)
    if app_state.status_bar is not None:
        app_state.status_bar.set_video_progress(progress)


def _drain_status_queue():
    if app_state.process_status_q is None:
        return
    while True:
        try:
            app_state.process_status_q.get_nowait()
        except Empty:
            return


def _kill_upscale_process_tree(process):
    """Hard-stop fallback after the worker's checkpoint grace period."""
    try:
        for child in psutil_Process(process.pid).children(recursive=True):
            try:
                child.kill()
            except (NoSuchProcess, AccessDenied):
                pass
    except (NoSuchProcess, AccessDenied):
        pass
    try:
        process.kill()
    except ProcessLookupError:
        pass  # It exited between the liveness check and kill.


def stop_upscale_process():
    """Blocking safety cleanup, used only after Tk's event loop has ended."""
    app_state.stopping = True
    event = app_state.event_stop_upscale_process
    if event is not None:
        event.set()
    process = app_state.process_upscale_orchestrator
    try:
        if process is not None and process.pid is not None:
            timeout = 10 if app_state.stop_deadline is None else max(
                0, app_state.stop_deadline - monotonic()
            )
            process.join(timeout=timeout)
            if process.is_alive():
                _kill_upscale_process_tree(process)
                process.join()
        app_state.process_upscale_orchestrator = None
        _drain_status_queue()
    finally:
        if event is not None:
            event.clear()
        app_state.processing = False
        app_state.processing_config = None
        app_state.stopping = False
        app_state.pending_upscale_status = None
        app_state.stop_deadline = None
        allow_sleep()


def _is_error_status(status):
    return isinstance(status, CacheSettingsConflict) or (
        isinstance(status, str) and status.startswith(ERROR_STATUS)
    )


def _finish_upscale(status: str | CacheSettingsConflict) -> bool:
    """Record a terminal result, but settle the UI only after the worker exits."""
    state = app_state
    process = state.process_upscale_orchestrator
    if not state.processing and process is None:
        return True  # A stale/duplicate completion must not notify twice.
    if state.pending_upscale_status is None or _is_error_status(status):
        state.pending_upscale_status = status
    if process is not None and process.pid is not None:
        if process.is_alive():
            if not state.stopping:
                state.info_message.set("Finalizing — waiting for worker cleanup...")
                if state.status_bar is not None:
                    state.status_bar.set_video_status("finalizing")
            return False
        process.join(timeout=0)
    status = state.pending_upscale_status
    if status == COMPLETED_STATUS and (process is None or process.exitcode != 0):
        exitcode = None if process is None else process.exitcode
        status = f"{ERROR_STATUS} Worker exited with code {exitcode} after completion"
    state.process_upscale_orchestrator = None
    _drain_status_queue()
    if state.event_stop_upscale_process is not None:
        state.event_stop_upscale_process.clear()
    state.pending_upscale_status = None
    state.stop_deadline = None
    config = state.processing_config
    state.processing_config = None
    allow_sleep()
    if state.closing:
        state.processing = False
        state.stopping = False
        _destroy_window()
        return True
    update_file_widget()
    if state.file_widget is not None:
        state.file_widget.clear_active_highlight()
    state.processing = False
    state.stopping = False
    App.place_upscale_button()
    if state.status_bar is not None:
        video_status = ("job finished" if status == COMPLETED_STATUS else
                        "stopped" if status == STOP_STATUS else "error")
        state.status_bar.set_video_status(video_status)
    if status == COMPLETED_STATUS:
        state.info_message.set("All files completed! :)")
        # Provider imports and PowerShell/toast delivery may block. This worker
        # never touches Tk or AppState and must not hold up window shutdown.
        Thread(
            target=show_completion_notification,
            name="QualityScaler-completion-notification", daemon=True,
        ).start()
    elif status == STOP_STATUS:
        state.info_message.set("Upscaling stopped")
    else:
        state.info_message.set("Error while upscaling :(")
        if (isinstance(status, CacheSettingsConflict) and config is not None
                and 1 <= status.file_number <= len(config.selected_file_list)):
            show_error_message(
                f"Video: {config.selected_file_list[status.file_number - 1]}\n"
                f"Failed run output folder: {config.selected_output_path}\n\n{status.message}",
                clear_cache_command=lambda use_failed_settings: _retry_after_cache_conflict(
                    config, status.file_number, use_failed_settings=use_failed_settings,
                ),
            )
        else:
            message = status.message if isinstance(status, CacheSettingsConflict) else (
                status.removeprefix(ERROR_STATUS).strip()
            )
            show_error_message(message)
    return True


def check_upscale_steps():
    _cancel_after(app_state.poll_after_id)
    app_state.poll_after_id = None
    if not app_state.processing:
        return
    process = app_state.process_upscale_orchestrator
    worker_alive = process is not None and process.pid is not None and process.is_alive()
    try:
        actual_step = app_state.process_status_q.get_nowait()
    except Empty:
        actual_step = None
    if _is_error_status(actual_step) or actual_step in (COMPLETED_STATUS, STOP_STATUS):
        if _finish_upscale(actual_step):
            return
    elif (actual_step is not None and actual_step != CLOSE_APP_STATUS
          and app_state.pending_upscale_status is None
          and not app_state.stopping and not app_state.closing):
        _display_status(actual_step)
        message = actual_step.message if isinstance(actual_step, VideoProgress) else actual_step
        try:
            file_number = int(message.split(".")[0])
        except (ValueError, IndexError):
            pass
        else:
            if app_state.file_widget is not None:
                app_state.file_widget.highlight_active_file(file_number)
    if not worker_alive:
        # Manager queues are synchronous: an exited process has already
        # delivered its final status. Observe exit BEFORE reading the queue so
        # a worker finishing between those checks cannot look like a crash.
        if app_state.pending_upscale_status is not None or actual_step is None:
            status = app_state.pending_upscale_status or (
                STOP_STATUS if app_state.stopping else
                f"{ERROR_STATUS} Worker exited without a completion status"
            )
            _finish_upscale(status)
            return
    elif app_state.stop_deadline is not None and monotonic() >= app_state.stop_deadline:
        _kill_upscale_process_tree(process)
        app_state.stop_deadline = None
    if app_state.closing:
        app_state.poll_after_id = _schedule(250, check_upscale_steps, during_close=True)
    else:
        app_state.poll_after_id = _schedule(250, check_upscale_steps)


def stop_button_command():
    """Request a checkpoint, then poll instead of joining on the Tk thread."""
    if not app_state.processing and app_state.process_upscale_orchestrator is None:
        return
    app_state.processing = True
    if not app_state.stopping:
        app_state.stopping = True
        app_state.stop_deadline = monotonic() + 10
        if app_state.event_stop_upscale_process is not None:
            app_state.event_stop_upscale_process.set()
    if not _is_error_status(app_state.pending_upscale_status):
        app_state.pending_upscale_status = STOP_STATUS
    if not app_state.closing:
        app_state.info_message.set("Stopping — saving progress and cleaning up...")
        if app_state.status_bar is not None:
            app_state.status_bar.set_video_status("stopping")
    check_upscale_steps()


def upscale_button_command():
    if inputs_are_busy():
        return
    config = build_processing_config()
    if config is None:
        return
    _start_upscale(config)


def _video_cache_directory(config, file_number):
    return Path(prepare_output_video_directory_name(
        config.selected_file_list[file_number - 1], config.selected_output_path,
        config.selected_AI_model, config.input_resize_factor,
        config.output_resize_factor, config.selected_sharpening_amount,
    )).resolve()


def _retry_after_cache_conflict(config, file_number, *, use_failed_settings=False):
    if app_state.closing or inputs_are_busy():
        return
    try:
        retry_config = config if use_failed_settings else _processing_config(config.selected_file_list)
    except ValueError as error:
        app_state.info_message.set(str(error))
        return
    warning = check_disk_space(
        retry_config.selected_output_path, retry_config.selected_file_list[file_number - 1:],
    )
    if warning is not None:
        app_state.info_message.set("Not enough disk space")
        show_disk_space_error_message(warning)
        return
    # Consent applies only to the reported cache, not a cache in a newly chosen
    # output folder. A different folder can continue without deleting either.
    clear_cache = (
        _video_cache_directory(config, file_number)
        == _video_cache_directory(retry_config, file_number)
    )
    _start_upscale(retry_config, retry_from=file_number, clear_cache=clear_cache)


def _start_upscale(config, *, retry_from=None, clear_cache=False):
    """Keep batch position separate from the explicit permission to reset a cache."""
    close_upscale_preview()  # Release VRAM for both normal starts and cache retries.
    _drain_status_queue()
    app_state.event_stop_upscale_process.clear()
    app_state.pending_upscale_status = None
    app_state.processing_config = config
    app_state.stop_deadline = None
    app_state.stopping = False
    _display_status("Loading")
    app_state.processing = True
    App.place_stop_button()
    app_state.process_upscale_orchestrator = Process(
        target=upscale_orchestrator,
        args=(
            app_state.process_status_q,
            app_state.event_stop_upscale_process,
            config.selected_file_list,
            config.selected_output_path,
            config.selected_AI_model,
            config.selected_AI_multithreading,
            config.input_resize_factor,
            config.output_resize_factor,
            config.selected_gpu,
            config.tiles_resolution,
            config.selected_sharpening_amount,
            config.selected_keep_frames,
            config.selected_image_extension,
            config.selected_video_extension,
            config.selected_video_codec,
            config.selected_video_options,
        ),
        **({"kwargs": {"retry_from": retry_from, "clear_cache": clear_cache}}
           if retry_from is not None else {}),
    )
    try:
        prevent_sleep()
        app_state.process_upscale_orchestrator.start()
    except Exception as error:
        _finish_upscale(f"{ERROR_STATUS} {error}")
        return
    app_state.poll_after_id = _schedule(250, check_upscale_steps)


# Preferences and lifecycle --------------------------------------------------

_PREFERENCE_FIELDS = {
    "app_zoom": "default_app_zoom",
    "ai_model": "default_AI_model",
    "ai_multithreading": "default_AI_multithreading",
    "gpu": "default_gpu",
    "keep_frames": "default_keep_frames",
    "image_extension": "default_image_extension",
    "video_extension": "default_video_extension",
    "video_codec": "default_video_codec",
    "video_preset": "default_video_preset",
    "video_rate_control": "default_video_rate_control",
    "video_crf": "default_video_crf",
    "video_bitrate": "default_video_bitrate",
    "sharpening": "default_sharpening",
    "output_path": "default_output_path",
    "input_resize_factor": "default_input_resize_factor",
    "output_resize_factor": "default_output_resize_factor",
    "vram_limiter": "default_VRAM_limiter",
}


def save_user_choices_in_json():
    preferences = app_state.preferences
    preferences.output_path = app_state.selected_output_path.get()
    preferences.input_resize_factor = str(app_state.selected_input_resize_factor.get())
    preferences.output_resize_factor = str(app_state.selected_output_resize_factor.get())
    preferences.vram_limiter = str(app_state.selected_VRAM_limiter.get())
    if app_state.selected_video_crf is not None:
        preferences.video_crf = app_state.selected_video_crf.get()
    if app_state.selected_video_bitrate is not None:
        preferences.video_bitrate = app_state.selected_video_bitrate.get()
    values = {key: getattr(preferences, name) for name, key in _PREFERENCE_FIELDS.items()}
    values["default_keep_frames"] = "ON" if preferences.keep_frames else "OFF"
    with open(USER_PREFERENCE_PATH, "w", encoding="utf-8") as preference_file:
        preference_file.write(json_dumps(values))


def load_user_preferences():
    try:
        with open(USER_PREFERENCE_PATH, "r", encoding="utf-8") as preference_file:
            values = json_load(preference_file)
        if not isinstance(values, dict):
            return UserPreferences()
    except (OSError, ValueError):
        return UserPreferences()
    preferences = UserPreferences()
    choices = dict(app_zoom=zoom_option_list, ai_model=MODEL_NAMES,
                   ai_multithreading=AI_multithreading_list, sharpening=sharpening_list,
                   image_extension=image_extension_list, video_extension=video_extension_list,
                   video_codec=[codec for codec in video_codec_list if codec != "----"])
    for name, key in _PREFERENCE_FIELDS.items():
        value = values.get(key)
        if name == "keep_frames":
            if value in ("ON", "OFF"):
                preferences.keep_frames = value == "ON"
        elif isinstance(value, str) and (name not in choices or value in choices[name]):
            setattr(preferences, name, value)
    _normalize_encoder_preferences(preferences)
    return preferences


def _destroy_window():
    try:
        close_upscale_preview()
    except TclError:
        pass  # A host may already have destroyed Tk's children.
    for handle in tuple(app_state.after_ids):
        _cancel_after(handle)
    app_state.poll_after_id = None
    app_state.preview_after_id = None
    for variable, handle in app_state.traces:
        try:
            variable.trace_remove("write", handle)
        except TclError:
            pass
    app_state.traces.clear()
    if app_state.file_widget is not None:
        widget = app_state.file_widget
        app_state.file_widget = None
        try:
            widget.destroy()
        except TclError:
            # Tk can already have destroyed every child when the root was
            # closed directly by a host, an OS callback, or a smoke test.
            pass
    if app_state.window is not None:
        window = app_state.window
        app_state.window = None
        try:
            # CTk also schedules appearance, DPI and textbox-scrollbar timers
            # outside our registry. Cancel this interpreter's remaining timers,
            # leaving command deletion to each owning widget's destroy method.
            for handle in window.tk.splitlist(window.tk.call("after", "info")):
                window.tk.call("after", "cancel", handle)
        except TclError:
            pass
        try:
            window.grab_release()
        except TclError:
            pass
        try:
            window.destroy()
        except TclError:
            pass
    app_state.input_widgets = ()
    app_state.action_button = None
    app_state.input_panel = None
    app_state.settings_panel = None
    app_state.settings_content = None
    app_state.header = None
    app_state.action_panel = None
    app_state.status_bar = None
    app_state.option_rows.clear()
    App._message_label = None
    App._message_after_id = None


def on_app_close():
    if app_state.closing:
        return
    try:
        save_user_choices_in_json()
    except OSError as error:
        # A missing/unwritable Documents folder must not prevent shutdown.
        print(f"[{app_name}] Could not save preferences: {error}")
    app_state.closing = True
    close_upscale_preview()
    stop_button_command()
    if not app_state.processing and app_state.window is not None:
        _destroy_window()


def place_option_background(row):
    if row not in app_state.option_rows:
        background = CTkFrame(
            app_state.settings_content, fg_color=CARD_BACKGROUND_COLOR,
            corner_radius=12, border_width=1, border_color=CARD_BORDER_COLOR,
        )
        background.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 8))
        background.grid_columnconfigure((0, 1), weight=1, uniform="options")
        app_state.option_rows[row] = background
    return app_state.option_rows[row]


class App:
    def __init__(self, window):
        window.protocol("WM_DELETE_WINDOW", on_app_close)
        window.title(get_AI_engine_info())
        window.geometry("1100x760")
        window.minsize(1000, 540)
        window.resizable(True, True)
        if sys.platform == "win32":
            window.iconbitmap(find_by_relative_path(f"Assets{os_separator}logo.ico"))
        window.grid_columnconfigure((0, 1), weight=1, uniform="panes")
        window.grid_rowconfigure(0, weight=1)
        app_state.input_panel = CTkFrame(window, fg_color=background_color, corner_radius=0)
        app_state.input_panel.grid(row=0, column=0, sticky="nsew")
        app_state.input_panel.grid_columnconfigure(0, weight=1)
        app_state.input_panel.grid_rowconfigure((0, 1), weight=1)
        app_state.settings_panel = CTkFrame(window, fg_color=background_color, corner_radius=0)
        app_state.settings_panel.grid(row=0, column=1, sticky="nsew")
        app_state.settings_panel.grid_columnconfigure(0, weight=1)
        app_state.settings_panel.grid_rowconfigure(1, weight=1)
        app_state.header = CTkFrame(app_state.settings_panel, fg_color=background_color)
        app_state.header.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        app_state.header.grid_columnconfigure(0, weight=1)
        app_state.settings_content = CTkScrollableFrame(
            app_state.settings_panel, fg_color=background_color, corner_radius=0, width=1,
        )
        app_state.settings_content.grid(row=1, column=0, sticky="nsew")
        app_state.settings_content.grid_columnconfigure(0, weight=1)
        app_state.action_panel = CTkFrame(app_state.settings_panel, fg_color=background_color)
        app_state.action_panel.grid(row=2, column=0, sticky="ew", padx=12, pady=12)
        app_state.action_panel.grid_columnconfigure(0, weight=1)
        self.place_loadFile_section()
        self.place_app_name()
        self.place_app_zoom_and_links()
        self.place_preview_link()
        self.place_AI_menu()
        self.place_AI_sharpening_menu()
        self.place_AI_multithreading_menu()
        self.place_input_output_resolution_textboxs()
        self.place_gpu_gpuVRAM_menus()
        self.place_image_video_output_menus()
        self.place_video_codec_keep_frames_menus()
        self.place_video_encoder_menus()
        self.place_output_path_textbox()
        self.place_message_label()
        self.place_upscale_button()

    @staticmethod
    def place_loadFile_section():
        for widget in app_state.input_widgets:
            widget.destroy()
        label = CTkLabel(
            app_state.input_panel,
            text=("DROP INPUT FILES HERE\n\nSUPPORTED FILES\n\n"
                  "IMAGES - jpg png tif bmp webp heic\n"
                  "VIDEOS - mp4 wmv webm mkv flv gif avi mov mpg qt 3gp"),
            fg_color=background_color, bg_color=background_color,
            text_color=CARD_MUTED_COLOR, width=300, height=150,
            font=bold13, anchor="center",
        )
        button = CTkButton(
            app_state.input_panel, command=open_files_action, text="SELECT FILES",
            width=140, height=30, font=bold12, border_width=2,
            corner_radius=UI_CORNER_RADIUS, fg_color="#282828",
            text_color="#E0E0E0", border_color=UI_ACCENT_COLOR,
        )
        label.grid(row=0, column=0, sticky="s", padx=12, pady=20)
        button.grid(row=1, column=0, sticky="n", pady=20)
        app_state.input_widgets = (label, button)

    @staticmethod
    def place_app_name():
        label = CTkLabel(
            app_state.header, text=f"{app_name} {version}",
            fg_color=background_color, bg_color=background_color,
            text_color=app_name_color, font=bold18, anchor="w",
        )
        label.grid(row=0, column=0, sticky="w")

    @staticmethod
    def place_app_zoom_and_links():
        label = CTkLabel(
            app_state.header, text="Zoom", width=35, height=22,
            fg_color="transparent", bg_color=background_color,
            text_color=CARD_TITLE_COLOR, font=bold13, anchor="w",
        )
        menu = App.create_option_menu(
            App.select_app_zoom, zoom_option_list, app_state.preferences.app_zoom,
            width=71, master=app_state.header,
        )
        label.grid(row=0, column=1, padx=(8, 4))
        menu.grid(row=0, column=2, padx=(0, 8))
        telegram = App.create_link_button(
            lambda: open_browser(telegramme, new=1), logo_telegram, master=app_state.header,
        )
        github = App.create_link_button(
            lambda: open_browser(githubme, new=1), logo_git, master=app_state.header,
        )
        telegram.grid(row=0, column=3, padx=(0, 6))
        github.grid(row=0, column=4)

    @staticmethod
    def place_preview_link():
        link = CTkButton(
            app_state.header, text="Preview upscale ↗", command=open_upscale_preview,
            fg_color="transparent", hover_color=widget_background_color,
            text_color=UI_ACCENT_COLOR, font=bold13, anchor="w",
            width=150, height=28,
        )
        link.grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))

    @staticmethod
    def _single_option(row, title, subtitle, help_text, command, choices, value):
        background = place_option_background(row)
        info = App.create_info_button(
            lambda: open_info_messagebox(title, subtitle, help_text), title, master=background,
        )
        menu = App.create_option_menu(command, choices, value, master=background)
        info.grid(row=0, column=0, sticky="w", padx=12, pady=10)
        menu.grid(row=0, column=1, sticky="ew", padx=(8, 12), pady=10)

    @staticmethod
    def place_AI_menu():
        App._single_option(
            ROW_AI_MODEL, "AI model", "Select the AI model based on your content type",
            [
                "\nLVAx2\n - Target: Live-action video upscaling\n - Tips: AI sharpening - OFF/Low\n",
                "\nRealESR_Gx4 - RealESR_Ax4\n - Target: Animated/degraded live-action video upscaling\n"
                " - Tips: AI sharpening - Low for animation, High for live-action videos\n",
                "\nBSRGANx2 - BSRGANx4\n - Target: High-quality image upscaling\n"
                " - Tips: can be used to upscale videos but will be slow\n",
                "\nRealESRGANx4\n - Target: High-quality image upscaling\n"
                " - Tips: can be used to upscale videos but will be slow\n",
                "\nMSharpx4\n - Target: Image upscaling and sharpening\n"
                " - Tips: to use on good quality photos (not too much noise)\n",
                "\nIRCNN_Mx1 - IRCNN_Lx1\n - Target: Video/image denoising\n - Tips: AI sharpening - OFF\n",
            ], App.select_AI_from_menu, AI_models_list, app_state.preferences.ai_model,
        )

    @staticmethod
    def place_AI_sharpening_menu():
        App._single_option(
            ROW_AI_SHARPENING, "AI sharpening", "Sharpen the AI output to fine-tune detail",
            [
                " Sharpening applies an unsharp mask to the AI result to bring back perceived detail",
                "\n SHARPENING OPTIONS\n - [OFF] No sharpening is applied\n"
                " - [Low] Light sharpening\n - [High] Stronger sharpening\n",
                "\n NOTES\n - Useful when the AI output looks too soft or smooth\n"
                " - Works only on the AI output, so it can't reintroduce source noise/compression artifacts\n",
            ], App.select_sharpening_from_menu, sharpening_list, app_state.preferences.sharpening,
        )

    @staticmethod
    def place_AI_multithreading_menu():
        App._single_option(
            ROW_AI_MULTITHREADING, "AI multithreading",
            "Process multiple video frames in parallel to speed up upscaling",
            [
                " This option can enhance video upscaling performance, especially on powerful GPUs.",
                "\n AI MULTITHREADING OPTIONS\n - OFF - Processes one frame at a time.\n"
                " - 2 / 4 / 6 / 8 threads - Processes that many frames simultaneously.\n",
                "\n NOTES\n - Higher thread counts increase CPU, GPU, and RAM usage.\n"
                " - Monitor your system's temperature to prevent overheating.\n"
                " - If the chosen thread count exceeds GPU capacity, the app automatically selects an optimal value.\n",
            ], App.select_AI_multithreading_from_menu, AI_multithreading_list,
            app_state.preferences.ai_multithreading,
        )

    @staticmethod
    def _paired_option(row, left, title, subtitle, help_text, create_widget):
        background = place_option_background(row)
        slot = CTkFrame(background, fg_color="transparent", width=1)
        slot.grid(row=0, column=0 if left else 1, sticky="ew", padx=10, pady=10)
        slot.grid_columnconfigure(1, weight=1)
        info = App.create_info_button(
            lambda: open_info_messagebox(title, subtitle, help_text), title, master=slot,
        )
        info.grid(row=0, column=0, sticky="w", padx=(0, 8))
        create_widget(slot).grid(row=0, column=1, sticky="ew")

    @staticmethod
    def place_input_output_resolution_textboxs():
        for left, title, variable, subtitle, help_text in (
            (True, "Input scale %", App.get_input_resize_factor_var(),
             "Controls the resolution fed to the AI — lower is faster, higher is sharper",
             [" A high value (>50%) creates high quality photos/videos but is slower.",
              "\n For a 1080p (1920x1080) image/video\n"
              " - 25% => 480x270\n - 50% => 960x540\n - 75% => 1440x810\n - 100% => 1920x1080\n"]),
            (False, "Output scale %", App.get_output_resize_factor_var(),
             "Controls the final output resolution after AI upscaling",
             [" 100% keeps the exact resolution produced by the AI upscaling.",
              " A lower value downscales the AI result. A higher value increases size, not real detail.",
              "\n If the AI generates 4K (3840x2160)\n"
              " - 50% => 1920x1080\n - 100% => 3840x2160\n - 200% => 7680x4320 (interpolated)\n"]),
        ):
            App._paired_option(
                ROW_RESOLUTION, left, title, subtitle, help_text,
                lambda master, var=variable: App.create_text_box(
                    var, little_textbox_width, master=master,
                ),
            )

    @staticmethod
    def place_gpu_gpuVRAM_menus():
        choices = GPU.menu_list()
        if app_state.preferences.gpu not in choices:
            app_state.preferences.gpu = GPU.default()
        App._paired_option(
            ROW_GPU, True, "GPU", "Select which GPU to use for AI inference",
            [" The app automatically detects the GPUs installed on your system.",
             " If no GPU is detected the menu shows \"No GPU found\" and the app may fall back to the CPU.",
             " Keep GPU drivers up to date. A more powerful GPU speeds up upscaling."],
            lambda master: App.create_option_menu(
                App.select_gpu_from_menu, choices, app_state.preferences.gpu,
                width=little_menu_width, master=master,
            ),
        )
        App._paired_option(
            ROW_GPU, False, "GPU VRAM (GB)",
            "Match this value to your GPU's available VRAM to avoid out-of-memory errors",
            [" Auto-detected from the selected GPU's dedicated VRAM at startup.",
             " You can override an incorrect value manually.",
             " Setting a value above available VRAM may cause upscale failure.",
             " For Intel UHD / Iris Xe or AMD Vega 3/5/7 integrated GPUs, select 2 GB."],
            lambda master: App.create_text_box(
                App.get_vram_limiter_var(), little_textbox_width, master=master,
            ),
        )

    @staticmethod
    def place_image_video_output_menus():
        App._paired_option(
            ROW_OUTPUT_FORMAT, True, "Image ext.", "Choose the output format for upscaled images",
            ["\n PNG\n - Lossless, very good quality; slow and heavy\n - Supports transparency\n",
             "\n JPG\n - Lossy, good quality; fast and lightweight\n - No transparency\n",
             "\n BMP\n - Uncompressed, highest quality; large file size\n - No transparency\n",
             "\n TIFF\n - Highest quality; very slow and heavy\n - Supports transparency\n"],
            lambda master: App.create_option_menu(
                App.select_image_extension_from_menu, image_extension_list,
                app_state.preferences.image_extension, width=little_menu_width, master=master,
            ),
        )
        App._paired_option(
            ROW_OUTPUT_FORMAT, False, "Video ext.", "Choose the container format for upscaled videos",
            ["\n MP4\n - Widely supported, efficient compression; ideal for general use\n",
             "\n MKV\n - High quality; supports almost any codec and multiple audio/subtitle tracks\n",
             "\n AVI\n - Older, compatible format; larger files with less efficient compression\n",
             "\n MOV\n - High quality Apple format; large files, suited for editing\n"],
            lambda master: App.create_option_menu(
                App.select_video_extension_from_menu, video_extension_list,
                app_state.preferences.video_extension, width=little_menu_width, master=master,
            ),
        )

    @staticmethod
    def place_video_codec_keep_frames_menus():
        App._paired_option(
            ROW_CODEC, True, "Video codec", "Choose the encoder used to compress the upscaled video",
            ["\n SOFTWARE ENCODING (CPU)\n - x264 | H.264\n - x265 | HEVC (H.265)\n",
             "\n NVIDIA GPU (NVENC)\n - h264_nvenc | H.264\n - hevc_nvenc | HEVC\n",
             "\n AMD GPU (AMF)\n - h264_amf | H.264\n - hevc_amf | HEVC\n",
             "\n INTEL GPU (QSV)\n - h264_qsv | H.264\n - hevc_qsv | HEVC\n"],
            lambda master: App.create_option_menu(
                App.select_video_codec_from_menu, video_codec_list,
                app_state.preferences.video_codec, width=little_menu_width,
                variable=App.get_video_codec_var(), master=master,
            ),
        )
        App._paired_option(
            ROW_CODEC, False, "Keep frames", "Keep the SQLite frame cache after encoding",
            ["\n ON\n - Keep the video's .qualityscaler/frames.sqlite3 cache after creating the output\n",
             "\n OFF\n - Remove the frame cache after successful encoding\n"
             " - Interrupted jobs remain resumable\n"],
            lambda master: App.create_option_menu(
                App.select_save_frame_from_menu, keep_frames_list,
                "ON" if app_state.preferences.keep_frames else "OFF",
                width=little_menu_width, master=master,
            ),
        )

    @staticmethod
    def place_video_encoder_menus():
        preferences = app_state.preferences
        _normalize_encoder_preferences(preferences)
        capabilities = encoder_capabilities(preferences.video_codec)
        App._single_option(
            ROW_ENCODER_PRESET, "Encoder preset", "Trade encoding speed for compression efficiency",
            [" Default lets the encoder choose its own preset.",
             " x264/x265: ultrafast is fastest; slower presets usually create smaller files at the same CRF.",
             " NVIDIA: p1 is fastest, p7 is slowest with higher quality.",
             " AMD: speed, balanced, or quality. Intel: veryfast through veryslow.",
             " Only presets supported by the selected encoder are listed."],
            App.select_video_preset_from_menu, list(capabilities.presets), preferences.video_preset,
        )
        App._paired_option(
            ROW_ENCODER_QUALITY, True, "Rate control", "Choose quality-based or bitrate-based encoding",
            [" CRF is available for x264/x265 only. Lower values mean higher quality and larger files.",
             " Bitrate sets a target video bitrate, not a guaranteed file size or constant bitrate.",
             " Hardware encoders do not expose CRF; their quality modes are not equivalent to CRF."],
            lambda master: App.create_option_menu(
                App.select_video_rate_control_from_menu, list(capabilities.rate_controls),
                preferences.video_rate_control, width=little_menu_width, master=master,
            ),
        )
        crf = preferences.video_rate_control == "CRF"
        App._paired_option(
            ROW_ENCODER_QUALITY, False, "CRF" if crf else "Bitrate kb/s",
            "Set constant rate factor" if crf else "Set the target video bitrate in kilobits per second",
            ([" Range: 0–51, including decimals. Lower is better quality; 23 is a starting point.",
              " The same CRF number is not equivalent quality across different encoders."]
             if crf else [" Enter a positive whole number in kb/s; 50000 = 50 Mb/s.",
                          " Higher bitrates generally produce larger files and better quality."]),
            lambda master: App.create_text_box(
                app_state.selected_video_crf if crf else app_state.selected_video_bitrate,
                little_textbox_width, master=master,
            ),
        )

    @staticmethod
    def place_output_path_textbox():
        background = place_option_background(ROW_OUTPUT_PATH)
        background.grid_columnconfigure(0, weight=1)
        background.grid_columnconfigure(1, weight=0, uniform="")
        info = App.create_info_button(
            lambda: open_info_messagebox(
                "Output path", "Choose where upscaled files are saved",
                [" The default is the folder containing each input file.",
                 " Select another folder using the SELECT button."],
            ), "Output path", master=background,
        )
        entry = App.create_text_box(
            App.get_output_path_var(), 1, state=DISABLED, master=background,
        )
        button = App.create_active_button(
            open_output_path_action, "SELECT", None, 70, 26, master=background,
        )
        info.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))
        button.grid(row=0, column=1, padx=12, pady=(10, 6))
        entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))

    @staticmethod
    def place_message_label():
        bar = StatusBar(
            App.get_app_window(), textvariable=App.get_info_message_var(),
            font=bold12, fg_color=CARD_BACKGROUND_COLOR, text_color=UI_ACCENT_COLOR,
        )
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        app_state.status_bar = bar
        App._message_label = bar.label
        App._message_color = UI_ACCENT_COLOR
        variable = App.get_info_message_var()
        app_state.traces.append((variable, variable.trace_add("write", App._refresh_message_color)))

    @staticmethod
    def _refresh_message_color(*_args):
        label = getattr(App, "_message_label", None)
        if label is None or app_state.closing or not label.winfo_exists():
            return
        message = App.get_info_message_var().get().lower()
        if "error" in message:
            target = MESSAGE_ERROR_COLOR
        elif "completed" in message:
            target = MESSAGE_SUCCESS_COLOR
        elif "stopped" in message:
            target = MESSAGE_WARNING_COLOR
        else:
            target = UI_ACCENT_COLOR
        App._animate_message_color(target)

    @staticmethod
    def _set_message_color(color):
        label = getattr(App, "_message_label", None)
        if label is not None and label.winfo_exists():
            label.configure(text_color=color)
        App._message_color = color

    @staticmethod
    def _animate_message_color(target):
        _cancel_after(getattr(App, "_message_after_id", None))
        App._message_after_id = None
        start = getattr(App, "_message_color", UI_ACCENT_COLOR)
        if start == target:
            App._set_message_color(target)
            return

        def run(step):
            if app_state.closing:
                return
            App._set_message_color(lerp_hex(start, target, min(1.0, step / 10)))
            if step < 10:
                App._message_after_id = _schedule(24, lambda: run(step + 1))
            else:
                App._message_after_id = None
        run(1)

    @staticmethod
    def _place_action(command, text, icon, border_color=UI_ACCENT_COLOR):
        if app_state.action_button is not None:
            app_state.action_button.destroy()
        button = App.create_active_button(
            command, text, icon, width=150, height=34, border_color=border_color,
            master=app_state.action_panel,
        )
        button.grid(row=0, column=0, sticky="ew")
        app_state.action_button = button

    @staticmethod
    def place_stop_button():
        App._place_action(stop_button_command, "STOP", stop_icon, "#EC1D1D")

    @staticmethod
    def place_upscale_button():
        App._place_action(upscale_button_command, "UPSCALE", upscale_icon)

    @staticmethod
    def select_app_zoom(value):
        app_state.preferences.app_zoom = value
        apply_app_zoom(float(value.replace("%", "")) / 100)

    @staticmethod
    def select_AI_from_menu(value):
        app_state.preferences.ai_model = value
        update_file_widget()

    @staticmethod
    def select_AI_multithreading_from_menu(value):
        app_state.preferences.ai_multithreading = value
        update_file_widget()

    @staticmethod
    def select_sharpening_from_menu(value):
        app_state.preferences.sharpening = value
        update_file_widget()

    @staticmethod
    def select_gpu_from_menu(value):
        app_state.preferences.gpu = value
        apply_auto_vram_for_gpu(value)
        apply_auto_codec_for_gpu(value)
        update_file_widget()

    @staticmethod
    def select_save_frame_from_menu(value):
        app_state.preferences.keep_frames = value == "ON"
        update_file_widget()

    @staticmethod
    def select_image_extension_from_menu(value):
        app_state.preferences.image_extension = value

    @staticmethod
    def select_video_extension_from_menu(value):
        app_state.preferences.video_extension = value
        update_file_widget()

    @staticmethod
    def select_video_codec_from_menu(value):
        if value == "----":
            app_state.selected_video_codec.set(app_state.preferences.video_codec)
            return
        app_state.preferences.video_codec = value
        refresh_encoder_controls()
        update_file_widget()

    @staticmethod
    def select_video_preset_from_menu(value):
        app_state.preferences.video_preset = value

    @staticmethod
    def select_video_rate_control_from_menu(value):
        app_state.preferences.video_rate_control = value
        refresh_encoder_controls()

    @staticmethod
    def create_info_button(command, text, *, master):
        frame = CTkFrame(master, fg_color=CARD_BACKGROUND_COLOR, width=1, height=25)
        button = CTkButton(
            frame, command=command, font=bold14, text="?", border_width=0,
            fg_color=CARD_BACKGROUND_COLOR, hover_color=CARD_BACKGROUND_COLOR,
            text_color=CARD_MUTED_COLOR, width=20, height=20,
            corner_radius=UI_CORNER_RADIUS,
        )
        button.bind("<Enter>", lambda e: button.configure(text_color=UI_ACCENT_COLOR))
        button.bind("<Leave>", lambda e: button.configure(text_color=CARD_MUTED_COLOR))
        button.grid(row=0, column=0, padx=(0, 7), pady=0, sticky="w")
        CTkLabel(
            frame, text=text, width=0, height=22, fg_color="transparent",
            bg_color=CARD_BACKGROUND_COLOR, text_color=CARD_VALUE_COLOR,
            font=bold13, anchor="w",
        ).grid(row=0, column=1, pady=0, sticky="w")
        frame.grid_rowconfigure(0, weight=1)
        return frame

    @staticmethod
    def create_option_menu(command, values, default_value,
                           width=159, height=28, variable=None, *, master):
        option_menu = CTkOptionMenu(
            master, command=command, values=values, variable=variable,
            width=width, height=height, corner_radius=UI_CORNER_RADIUS,
            dropdown_font=bold12, font=bold11, anchor="center",
            dynamic_resizing=False,
            text_color=text_color, fg_color=background_color,
            button_color=background_color, button_hover_color=background_color,
            dropdown_fg_color=background_color,
        )
        option_menu.set(default_value)
        return option_menu

    @staticmethod
    def create_text_box(textvariable, width, height=28, state="normal", *, master):
        return CTkEntry(
            master, textvariable=textvariable,
            corner_radius=UI_CORNER_RADIUS, width=width, height=height,
            font=bold11, justify="center", text_color=text_color,
            fg_color="#000000", border_width=2, border_color=UI_BORDER_COLOR,
            state=state,
        )

    @staticmethod
    def create_active_button(command, text, icon, width=140, height=30,
                             border_color=UI_ACCENT_COLOR, *, master):
        button = CTkButton(
            master, text=text, image=icon, width=width,
            height=height, font=bold11, border_width=2,
            corner_radius=UI_CORNER_RADIUS, fg_color="#282828",
            text_color="#E0E0E0", border_color=border_color,
        )

        def press_feedback():
            if button.winfo_exists():
                button.configure(fg_color="#1E1E1E")
                _schedule(120, lambda: button.winfo_exists() and button.configure(fg_color="#282828"))
            command()
        button.configure(command=press_feedback)
        return button

    @staticmethod
    def create_link_button(command, icon, *, master):
        button = CTkButton(
            master, command=command, image=icon,
            width=30, height=30, border_width=2, corner_radius=UI_CORNER_RADIUS,
            bg_color=background_color, fg_color=CARD_BACKGROUND_COLOR,
            hover_color=widget_background_color, text_color=text_color,
            border_color=CARD_BORDER_COLOR, anchor="center", text="", font=bold11,
        )
        button.bind("<Enter>", lambda e: button.configure(border_color=UI_ACCENT_COLOR))
        button.bind("<Leave>", lambda e: button.configure(border_color=CARD_BORDER_COLOR))
        return button

    @staticmethod
    def get_app_window():
        return app_state.window

    @staticmethod
    def get_info_message_var():
        return app_state.info_message

    @staticmethod
    def get_output_path_var():
        return app_state.selected_output_path

    @staticmethod
    def get_input_resize_factor_var():
        return app_state.selected_input_resize_factor

    @staticmethod
    def get_output_resize_factor_var():
        return app_state.selected_output_resize_factor

    @staticmethod
    def get_vram_limiter_var():
        return app_state.selected_VRAM_limiter

    @staticmethod
    def get_video_codec_var():
        return app_state.selected_video_codec


def _load_ui_resources():
    global bold11, bold12, bold13, bold14, bold17, bold18, bold22
    global logo_git, logo_telegram, stop_icon, upscale_icon, clear_icon
    bold11 = CTkFont(family="Segoe UI", size=11, weight="bold")
    bold12 = CTkFont(family="Segoe UI", size=12, weight="bold")
    bold13 = CTkFont(family="Segoe UI", size=13, weight="bold")
    bold14 = CTkFont(family="Segoe UI", size=14, weight="bold")
    bold17 = CTkFont(family="Segoe UI", size=17, weight="bold")
    bold18 = CTkFont(family="Segoe UI", size=18, weight="bold")
    bold22 = CTkFont(family="Segoe UI", size=22, weight="bold")

    def icon(name, size):
        with pillow_image_open(find_by_relative_path(f"Assets{os_separator}{name}")) as image:
            # Keep enough pixels for high-DPI/175% zoom without retaining the
            # original 1280–2048px artwork for a 15–18px toolbar icon.
            image.thumbnail((size[0] * 4, size[1] * 4))
            return CTkImage(image.copy(), size=size)
    logo_git = icon("github_logo.png", (18, 18))
    logo_telegram = icon("telegram_logo.png", (16, 16))
    stop_icon = icon("stop_icon.png", (15, 15))
    upscale_icon = icon("upscale_icon.png", (15, 15))
    clear_icon = icon("clear_icon.png", (15, 15))


def main():
    global app_state
    preferences = load_user_preferences()
    app_state = AppState(preferences=preferences)
    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")
    apply_app_zoom(float(preferences.app_zoom.replace("%", "")) / 100)
    manager = multiprocessing_Manager()
    try:
        app_state.process_status_q = manager.Queue(maxsize=1)
        app_state.event_stop_upscale_process = manager.Event()
        app_state.window = FileDropWindow()
        # Construct the complete interface before revealing it, instead of
        # showing an empty root followed by each individual canvas repaint.
        app_state.window.withdraw()
        for name in (
            "info_message", "selected_output_path", "selected_input_resize_factor",
            "selected_output_resize_factor", "selected_VRAM_limiter", "selected_video_codec",
            "selected_video_crf", "selected_video_bitrate",
        ):
            setattr(app_state, name, StringVar(master=app_state.window))
        update_splash("Detecting graphics adapters...")
        GPU.detect()
        if preferences.gpu not in GPU.menu_list():
            preferences.gpu = GPU.default()
        app_state.selected_input_resize_factor.set(preferences.input_resize_factor)
        app_state.selected_output_resize_factor.set(preferences.output_resize_factor)
        app_state.selected_output_path.set(preferences.output_path)
        app_state.selected_video_crf.set(preferences.video_crf)
        app_state.selected_video_bitrate.set(preferences.video_bitrate)
        apply_auto_vram_for_gpu(preferences.gpu, fallback=preferences.vram_limiter)
        apply_auto_codec_for_gpu(preferences.gpu)
        app_state.selected_video_codec.set(preferences.video_codec)
        app_state.info_message.set("Hi :)")
        for variable in (
            app_state.selected_input_resize_factor, app_state.selected_output_resize_factor,
            app_state.selected_output_path, app_state.selected_VRAM_limiter,
        ):
            app_state.traces.append((variable, variable.trace_add("write", schedule_file_widget_update)))
        update_splash("Preparing interface...")
        _load_ui_resources()
        App(app_state.window)
        app_state.window.file_drop_target = FileDropTarget(
            app_state.window, select_input_files, accepts=accept_input_drop,
            on_invalid=lambda: app_state.info_message.set("Unable to read the dropped file paths"),
        )
        app_state.window.deiconify()
        app_state.window.update()
        close_splash()
        app_state.window.mainloop()
    finally:
        try:
            # The root can also be destroyed externally. Keep the manager alive
            # through checkpoint/kill cleanup even when mainloop ends abruptly.
            app_state.closing = True
            try:
                stop_upscale_process()
            finally:
                _destroy_window()
        finally:
            manager.shutdown()
