
# Standard library imports
import sys
from time       import sleep
from webbrowser import open as open_browser

from typing    import Callable
from threading import Thread
from queue     import Empty
from multiprocessing import (
    Process,
    Event          as multiprocessing_Event,
    Manager        as multiprocessing_Manager,
    freeze_support as multiprocessing_freeze_support
)

from json import (
    load  as json_load,
    dumps as json_dumps
)

from os import (
    sep        as os_separator,
    devnull    as os_devnull
)

from os.path import (
    basename   as os_path_basename,
    join       as os_path_join,
    exists     as os_path_exists,
    expanduser as os_path_expanduser
)

# Third-party library imports
from psutil import Process as psutil_Process, NoSuchProcess, AccessDenied
from onnxruntime import get_available_providers as onnxruntime_get_available_providers

from PIL.Image import (
    open      as pillow_image_open,
    fromarray as pillow_image_fromarray
)

from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    error        as opencv_error,
    VideoCapture as opencv_VideoCapture,
    cvtColor     as opencv_cvtColor,
    resize       as opencv_resize,
)

from qualityscaler.ffmpeg import FFmpeg
from qualityscaler.drag_drop import FileDropTarget, FileDropWindow
from qualityscaler.inputs import merge_input_files, supported_input_files
from qualityscaler.jobs import check_if_file_is_video, upscale_orchestrator
from qualityscaler.media import (
    get_image_resolution,
    image_read,
)
from qualityscaler.paths import (
    OUTPUT_PATH_CODED,
    find_by_relative_path,
)
from qualityscaler.status import write_process_status

# GUI imports
from tkinter import StringVar
from tkinter import DISABLED
from customtkinter import (
    CTkFrame,
    CTkButton,
    CTkEntry,
    CTkFont,
    CTkImage,
    CTkLabel,
    CTkOptionMenu,
    CTkScrollableFrame,
    CTkToplevel,
    filedialog,
    set_appearance_mode,
    set_default_color_theme
)

if sys.stdout is None: sys.stdout = open(os_devnull, "w")
if sys.stderr is None: sys.stderr = open(os_devnull, "w")

app_name   = "QualityScaler"
version    = "4.7"
githubme   = "https://github.com/Djdefrag/QualityScaler/releases"
telegramme = "https://linktr.ee/j3ngystudio"

app_name_color          = "#F274EE"
background_color        = "#000000"
widget_background_color = "#181818"
text_color              = "#B8B8B8"

VRAM_model_usage = {
    'RealESR_Gx4':     2.5,
    'RealESR_Animex4': 2.5,
    'BSRGANx4':        0.75,
    'RealESRGANx4':    0.75,
    'BSRGANx2':        0.8,
    'IRCNN_Mx1':       4,
    'IRCNN_Lx1':       4,

    'LVAx2': 2.5,
    'LVAx4': 2.5
}

MENU_LIST_SEPARATOR         = [ "----" ]
SRVGGNetCompact_models_list = [ "RealESR_Gx4", "RealESR_Animex4"                ]
BSRGAN_models_list          = [ "BSRGANx2",    "BSRGANx4",       "RealESRGANx4" ]
IRCNN_models_list           = [ "IRCNN_Mx1",   "IRCNN_Lx1"                      ]
in_testing_model_list       = [ "LVAx2",       "LVAx4"                          ]

AI_models_list = (
    SRVGGNetCompact_models_list
    + MENU_LIST_SEPARATOR + BSRGAN_models_list
    + MENU_LIST_SEPARATOR + IRCNN_models_list
    #+ MENU_LIST_SEPARATOR + in_testing_model_list
    )

AI_multithreading_list = [ "OFF", "2 threads", "4 threads", "6 threads", "8 threads"]
blending_list          = [ "OFF", "Low", "Medium", "High" ]
gpus_list              = [ "Auto", "GPU 1", "GPU 2", "GPU 3", "GPU 4" ]
keep_frames_list       = [ "OFF", "ON" ]
image_extension_list   = [ ".png", ".jpg", ".bmp", ".tiff" ]
video_extension_list   = [ ".mp4", ".mkv", ".avi", ".mov" ]
video_codec_list = [
    "x264",       "x265",       MENU_LIST_SEPARATOR[0],
    "h264_nvenc", "hevc_nvenc", MENU_LIST_SEPARATOR[0],
    "h264_amf",   "hevc_amf",   MENU_LIST_SEPARATOR[0],
    "h264_qsv",   "hevc_qsv",
    ]

DOCUMENT_PATH        = os_path_join(os_path_expanduser('~'), 'Documents')
USER_PREFERENCE_PATH = find_by_relative_path(f"{DOCUMENT_PATH}{os_separator}{app_name}_{version}_UserPreference.json")
FFMPEG_EXE_PATH      = FFmpeg().executable

COMPLETED_STATUS = "Completed"
ERROR_STATUS     = "Error"
STOP_STATUS      = "Stop"


offset_y_options = 0.0825
row1  = 0.125
row2  = row1 + offset_y_options
row3  = row2 + offset_y_options
row4  = row3 + offset_y_options
row5  = row4 + offset_y_options
row6  = row5 + offset_y_options
row7  = row6 + offset_y_options
row8  = row7 + offset_y_options
row9  = row8 + offset_y_options
row10 = row9 + offset_y_options

column_offset = 0.2
column_info1  = 0.625
column_info2  = 0.858
column_1      = 0.66
column_2      = column_1 + column_offset
column_1_5    = column_info1 + 0.08
column_1_4    = column_1_5 - 0.0127
column_3      = column_info2 + 0.08
column_2_9    = column_3 - 0.0127
column_3_5    = column_2 + 0.0355

little_textbox_width = 74
little_menu_width = 98



file_widget = None
input_widgets = ()



# GUI utils ---------------------------

class MessageBox(CTkToplevel):

    def __init__(
            self,
            messageType: str,
            title: str,
            subtitle: str,
            default_value: str,
            option_list: list,
            ) -> None:

        super().__init__()

        self._running: bool = False

        self._messageType = messageType
        self._title       = title
        self._subtitle    = subtitle
        self._default_value = default_value
        self._option_list   = option_list
        self._ctkwidgets_index = 0

        self.title('')
        self.lift()                          # lift window on top
        self.attributes("-topmost", True)    # stay on top
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(10, self._create_widgets)  # create widgets with slight delay, to avoid white flickering of background
        self.resizable(False, False)
        self.grab_set()                       # make other windows not clickable

    def _ok_event(
            self,
            event = None
            ) -> None:
        self.grab_release()
        self.destroy()

    def _on_closing(
            self
            ) -> None:
        self.grab_release()
        self.destroy()

    def createEmptyLabel(self) -> CTkLabel:
        return CTkLabel(
            master   = self,
            fg_color = "transparent",
            width    = 500,
            height   = 17,
            text     = ''
        )

    def placeInfoMessageTitleSubtitle(self) -> None:

        spacingLabel1 = self.createEmptyLabel()
        spacingLabel2 = self.createEmptyLabel()

        if self._messageType == "info":
            title_subtitle_text_color = "#3399FF"
        elif self._messageType == "error":
            title_subtitle_text_color = "#FF3131"

        titleLabel = CTkLabel(
            master     = self,
            width      = 500,
            anchor     = 'w',
            justify    = "left",
            fg_color   = "transparent",
            text_color = title_subtitle_text_color,
            font       = bold22,
            text       = self._title
            )

        if self._default_value != None:
            defaultLabel = CTkLabel(
                master     = self,
                width      = 500,
                anchor     = 'w',
                justify    = "left",
                fg_color   = "transparent",
                text_color = "#3399FF",
                font       = bold17,
                text       = f"Default: {self._default_value}"
                )

        subtitleLabel = CTkLabel(
            master     = self,
            width      = 500,
            anchor     = 'w',
            justify    = "left",
            fg_color   = "transparent",
            text_color = title_subtitle_text_color,
            font       = bold14,
            text       = self._subtitle
            )

        spacingLabel1.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 0, pady = 0, sticky = "ew")

        self._ctkwidgets_index += 1
        titleLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 0, sticky = "ew")

        if self._default_value != None:
            self._ctkwidgets_index += 1
            defaultLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 0, sticky = "ew")

        self._ctkwidgets_index += 1
        subtitleLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 0, sticky = "ew")

        self._ctkwidgets_index += 1
        spacingLabel2.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 0, pady = 0, sticky = "ew")

    def placeInfoMessageOptionsText(self) -> None:

        for option_text in self._option_list:
            optionLabel = CTkLabel(
                master        = self,
                width         = 600,
                height        = 45,
                anchor        = 'w',
                justify       = "left",
                text_color    = text_color,
                fg_color      = "#282828",
                bg_color      = "transparent",
                font          = bold13,
                text          = option_text,
                corner_radius = 10,
            )

            self._ctkwidgets_index += 1
            optionLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 4, sticky = "ew")

        spacingLabel3 = self.createEmptyLabel()

        self._ctkwidgets_index += 1
        spacingLabel3.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 0, pady = 0, sticky = "ew")

    def placeInfoMessageOkButton(
            self
            ) -> None:

        ok_button = CTkButton(
            master  = self,
            command = self._ok_event,
            text    = 'OK',
            width   = 125,
            font         = bold11,
            border_width = 1,
            fg_color     = "#282828",
            text_color   = "#E0E0E0",
            border_color = "#0096FF"
        )

        self._ctkwidgets_index += 1
        ok_button.grid(row = self._ctkwidgets_index, column = 1, columnspan = 1, padx = (10, 20), pady = (10, 20), sticky = "e")

    def _create_widgets(
            self
            ) -> None:

        self.grid_columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)

        self.placeInfoMessageTitleSubtitle()
        self.placeInfoMessageOptionsText()
        self.placeInfoMessageOkButton()

class FileWidget(CTkScrollableFrame):

    def __init__(
            self,
            master,
            selected_file_list,
            upscale_factor       = 1,
            input_resize_factor  = 0,
            output_resize_factor = 0,
            **kwargs
            ) -> None:

        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight = 1)

        self.file_list            = selected_file_list
        self.upscale_factor       = upscale_factor
        self.input_resize_factor  = input_resize_factor
        self.output_resize_factor = output_resize_factor

        self.index_row = 1
        self.ui_components = []
        self._icons = {}
        self._header_created = False
        self.unreadable_files = []
        self._create_widgets()

    def _destroy_(self) -> None:
        clear_input_files()

    def _create_widgets(self) -> None:
        if not self._header_created:
            self.add_clean_button()
            self._header_created = True
        readable = []
        self.unreadable_files = []
        for file_path in self.file_list:
            try:
                file_name_label, file_info_label = self.add_file_information(file_path)
            except (OSError, ValueError, opencv_error, ZeroDivisionError):
                self.unreadable_files.append(file_path)
                continue
            readable.append(file_path)
            self.ui_components.append(file_name_label)
            self.ui_components.append(file_info_label)
        self.file_list = readable

    def add_file_information(self, file_path) -> tuple:
        infos, icon = self.extract_file_info(file_path)

        # File name
        file_name_label = CTkLabel(
            self,
            text       = os_path_basename(file_path),
            font       = bold14,
            text_color = text_color,
            compound   = "left",
            anchor     = "w",
            padx       = 10,
            pady       = 5,
            justify    = "left",
        )
        file_name_label.grid(
            row    = self.index_row,
            column = 0,
            columnspan = 3,
            pady   = (0, 2),
            padx   = (3, 3),
            sticky = "w"
        )

        # File infos and icon
        file_info_label = CTkLabel(
            self,
            text       = infos,
            image      = icon,
            font       = bold12,
            text_color = text_color,
            compound   = "left",
            anchor     = "w",
            padx       = 10,
            pady       = 5,
            justify    = "left",
        )
        file_info_label.grid(
            row    = self.index_row + 1,
            column = 0,
            columnspan = 3,
            pady   = (0, 15),
            padx   = (3, 3),
            sticky = "w"
        )

        self.index_row += 2

        return file_name_label, file_info_label

    def add_clean_button(self) -> None:
        hint = CTkLabel(self, text="Drop more files here", font=bold12, text_color=text_color)
        hint.grid(row=0, column=0, padx=(3, 8), pady=(7, 7), sticky="w")
        add_button = CTkButton(
            self, text="ADD FILES", command=open_files_action, width=90, height=28,
            font=bold11, corner_radius=1, fg_color="#282828",
        )
        add_button.grid(row=0, column=1, padx=(0, 7), pady=(7, 7))

        button = CTkButton(
            master        = self,
            command       = self._destroy_,
            text          = "CLEAN",
            image         = clear_icon,
            width         = 90,
            height        = 28,
            font          = bold11,
            border_width  = 1,
            corner_radius = 1,
            fg_color      = "#282828",
            text_color    = "#E0E0E0",
            border_color  = "#0096FF"
        )

        button.grid(row = 0, column=2, pady=(7, 7), padx = (0, 7))




    def extract_file_icon(self, file_path) -> CTkImage:
        if file_path in self._icons:
            return self._icons[file_path]
        max_size = 60

        if check_if_file_is_video(file_path):
            video_cap   = opencv_VideoCapture(file_path)
            try:
                success, frame = video_cap.read()
                if not success or frame is None:
                    raise ValueError(f"Unable to read video: {file_path}")
                source_icon = opencv_cvtColor(frame, COLOR_BGR2RGB)
            finally:
                video_cap.release()
        else:
            source_icon = opencv_cvtColor(image_read(file_path), COLOR_BGR2RGB)

        ratio       = min(max_size / source_icon.shape[0], max_size / source_icon.shape[1])
        new_width   = int(source_icon.shape[1] * ratio)
        new_height  = int(source_icon.shape[0] * ratio)
        source_icon = opencv_resize(source_icon,(new_width, new_height))
        ctk_icon    = CTkImage(pillow_image_fromarray(source_icon, mode="RGB"), size = (new_width, new_height))

        self._icons[file_path] = ctk_icon
        return ctk_icon

    def extract_file_info(self, file_path) -> tuple:

        if check_if_file_is_video(file_path):
            cap          = opencv_VideoCapture(file_path)
            try:
                width        = round(cap.get(CAP_PROP_FRAME_WIDTH))
                height       = round(cap.get(CAP_PROP_FRAME_HEIGHT))
                num_frames   = int(cap.get(CAP_PROP_FRAME_COUNT))
                frame_rate   = cap.get(CAP_PROP_FPS)
                if frame_rate <= 0:
                    raise ValueError(f"Invalid video frame rate: {file_path}")
                duration     = num_frames/frame_rate
                minutes      = int(duration/60)
                seconds      = duration % 60
            finally:
                cap.release()

            file_icon  = self.extract_file_icon(file_path)
            file_infos = f"{minutes}m:{round(seconds)}s • {num_frames}frames • {width}x{height} \n"

            if self.input_resize_factor != 0 and self.output_resize_factor != 0 and self.upscale_factor != 0 :
                input_resized_height = int(height * (self.input_resize_factor/100))
                input_resized_width  = int(width * (self.input_resize_factor/100))

                upscaled_height = int(input_resized_height * self.upscale_factor)
                upscaled_width  = int(input_resized_width * self.upscale_factor)

                output_resized_height = int(upscaled_height * (self.output_resize_factor/100))
                output_resized_width  = int(upscaled_width * (self.output_resize_factor/100))

                file_infos += (
                    f"AI input ({self.input_resize_factor}%) ➜ {input_resized_width}x{input_resized_height} \n"
                    f"AI output (x{self.upscale_factor}) ➜ {upscaled_width}x{upscaled_height} \n"
                    f"Video output ({self.output_resize_factor}%) ➜ {output_resized_width}x{output_resized_height}"
                )

        else:
            height, width = get_image_resolution(image_read(file_path))
            file_icon     = self.extract_file_icon(file_path)

            file_infos = f"{width}x{height}\n"

            if self.input_resize_factor != 0 and self.output_resize_factor != 0 and self.upscale_factor != 0 :
                input_resized_height = int(height * (self.input_resize_factor/100))
                input_resized_width  = int(width * (self.input_resize_factor/100))

                upscaled_height = int(input_resized_height * self.upscale_factor)
                upscaled_width  = int(input_resized_width * self.upscale_factor)

                output_resized_height = int(upscaled_height * (self.output_resize_factor/100))
                output_resized_width  = int(upscaled_width * (self.output_resize_factor/100))

                file_infos += (
                    f"AI input ({self.input_resize_factor}%) ➜ {input_resized_width}x{input_resized_height} \n"
                    f"AI output (x{self.upscale_factor}) ➜ {upscaled_width}x{upscaled_height} \n"
                    f"Image output ({self.output_resize_factor}%) ➜ {output_resized_width}x{output_resized_height}"
                )

        return file_infos, file_icon


    # EXTERNAL FUNCTIONS

    def clean_file_list(self) -> None:
        self.index_row = 1
        for ui_component in self.ui_components:
            ui_component.destroy()
        self.ui_components = []

    def get_selected_file_list(self) -> list:
        return self.file_list

    def set_upscale_factor(self, upscale_factor) -> None:
        self.upscale_factor = upscale_factor

    def set_input_resize_factor(self, input_resize_factor) -> None:
        self.input_resize_factor = input_resize_factor

    def set_output_resize_factor(self, output_resize_factor) -> None:
        self.output_resize_factor = output_resize_factor



def get_values_for_file_widget() -> tuple:
    # Upscale factor
    upscale_factor = get_upscale_factor()

    # Input resolution %
    try:
        input_resize_factor = int(float(str(selected_input_resize_factor.get())))
    except:
        input_resize_factor = 0

    # Output resolution %
    try:
        output_resize_factor = int(float(str(selected_output_resize_factor.get())))
    except:
        output_resize_factor = 0

    return upscale_factor, input_resize_factor, output_resize_factor

def update_file_widget(a, b, c) -> None:
    if file_widget is None:
        return

    upscale_factor, input_resize_factor, output_resize_factor = get_values_for_file_widget()

    file_widget.clean_file_list()
    file_widget.set_upscale_factor(upscale_factor)
    file_widget.set_input_resize_factor(input_resize_factor)
    file_widget.set_output_resize_factor(output_resize_factor)
    file_widget._create_widgets()

def create_option_background():
    return CTkFrame(
        master   = window,
        bg_color = background_color,
        fg_color = widget_background_color,
        height   = 46,
        corner_radius = 10
    )

def create_info_button(
        command: Callable,
        text:    str,
        width:   int = 200
        ) -> CTkFrame:

    frame = CTkFrame(master = window, fg_color = widget_background_color, height = 25)

    button = CTkButton(
        master        = frame,
        command       = command,
        font          = bold12,
        text          = "?",
        border_color  = "#0096FF",
        border_width  = 1,
        fg_color      = widget_background_color,
        hover_color   = background_color,
        width         = 23,
        height        = 15,
        corner_radius = 1
    )
    button.grid(row=0, column=0, padx=(0, 7), pady=2, sticky="w")

    label = CTkLabel(
        master     = frame,
        text       = text,
        width      = width,
        height     = 22,
        fg_color   = "transparent",
        bg_color   = widget_background_color,
        text_color = text_color,
        font       = bold13,
        anchor     = "w"
    )
    label.grid(row=0, column=1, sticky="w")

    frame.grid_propagate(False)
    frame.grid_columnconfigure(1, weight=1)

    return frame

def create_option_menu(
        command:       Callable,
        values:        list,
        default_value: str,
        border_color:  str = "#404040",
        border_width:  int = 1,
        width:         int = 159,
        height:        int = 26
    ) -> CTkFrame:

    total_width  = (width + 2 * border_width)
    total_height = (height + 2 * border_width)

    frame = CTkFrame(
        master        = window,
        fg_color      = border_color,
        width         = total_width,
        height        = total_height,
        border_width  = 0,
        corner_radius = 1,
    )

    option_menu = CTkOptionMenu(
        master             = frame,
        command            = command,
        values             = values,
        width              = width,
        height             = height,
        corner_radius      = 0,
        dropdown_font      = bold12,
        font               = bold11,
        anchor             = "center",
        text_color         = text_color,
        fg_color           = background_color,
        button_color       = background_color,
        button_hover_color = background_color,
        dropdown_fg_color  = background_color
    )

    option_menu.place(x = (total_width - width) / 2, y = (total_height - height) / 2)
    option_menu.set(default_value)
    return frame

def create_text_box(
        textvariable: StringVar,
        width:        int,
        height:       int = 26
    ) -> CTkEntry:

    return CTkEntry(
        master        = window,
        textvariable  = textvariable,
        corner_radius = 1,
        width         = width,
        height        = height,
        font          = bold11,
        justify       = "center",
        text_color    = text_color,
        fg_color      = "#000000",
        border_width  = 1,
        border_color  = "#404040",
    )

def create_text_box_output_path(
        textvariable: StringVar,
        height:       int = 26
    ) -> CTkEntry:

    return CTkEntry(
        master        = window,
        textvariable  = textvariable,
        corner_radius = 1,
        width         = 250,
        height        = height,
        font          = bold11,
        justify       = "center",
        text_color    = text_color,
        fg_color      = "#000000",
        border_width  = 1,
        border_color  = "#404040",
        state         = DISABLED
    )

def create_active_button(
        command: Callable,
        text: str,
        icon: CTkImage = None,
        width: int = 140,
        height: int = 30,
        border_color: str = "#0096FF"
        ) -> CTkButton:

    return CTkButton(
        master        = window,
        command       = command,
        text          = text,
        image         = icon,
        width         = width,
        height        = height,
        font          = bold11,
        border_width  = 1,
        corner_radius = 1,
        fg_color      = "#282828",
        text_color    = "#E0E0E0",
        border_color  = border_color
    )




# Core functions ------------------------

def check_upscale_steps() -> None:
    sleep(1)

    while True:
        actual_step = process_status_q.get()
        print(f"[{app_name}] check_upscale_steps - {actual_step}")

        if actual_step == COMPLETED_STATUS:
            info_message.set(f"All files completed! :)")
            stop_upscale_process()
            place_upscale_button()
            break

        elif actual_step == STOP_STATUS:
            info_message.set(f"Upscaling stopped")
            stop_upscale_process()
            place_upscale_button()
            break

        elif ERROR_STATUS in actual_step:
            info_message.set(f"Error while upscaling :(")
            error_to_show = actual_step.replace(ERROR_STATUS, "")
            show_error_message(error_to_show.strip())
            place_upscale_button()
            break

        else:
            info_message.set(actual_step)

        sleep(1)

def stop_upscale_process() -> None:
    """Checkpoint normally first; a hung GPU must not keep the app alive."""
    event_stop_upscale_process.set()
    process = globals().get("process_upscale_orchestrator")
    if process is not None and process.pid is not None:
        process.join(timeout=10)
        if process.is_alive():
            # FFmpeg is a child of the orchestrator. Do not orphan it when a
            # blocked model requires the hard-stop fallback.
            try:
                for child in psutil_Process(process.pid).children(recursive=True):
                    try:
                        child.kill()
                    except (NoSuchProcess, AccessDenied):
                        pass
            except (NoSuchProcess, AccessDenied):
                pass
            process.kill()
            process.join()
    while True:
        try:
            process_status_q.get_nowait()
        except Empty:
            break
    event_stop_upscale_process.clear()

def stop_button_command() -> None:
    stop_upscale_process()
    write_process_status(process_status_q, f"{STOP_STATUS}")

# ORCHESTRATOR

def upscale_button_command() -> None:
    global selected_file_list
    global selected_AI_model
    global selected_gpu
    global selected_keep_frames
    global selected_AI_multithreading
    global selected_blending_factor
    global selected_image_extension
    global selected_video_extension
    global selected_video_codec
    global tiles_resolution
    global input_resize_factor
    global output_resize_factor

    global process_upscale_orchestrator

    if user_input_checks():
        info_message.set("Loading")

        print("=" * 50)
        print("> Starting upscale:")
        print(f"    Files to upscale: {len(selected_file_list)}")
        print(f"    Output path: {(selected_output_path.get())}")
        print(f"    Selected AI model: {selected_AI_model}")
        print(f"    Blending factor: {selected_blending_factor}")
        print(f"    AI multithreading: {selected_AI_multithreading}")
        print(f"    Selected GPU: {selected_gpu}")
        print(f"    Tiles resolution for selected GPU VRAM: {tiles_resolution}x{tiles_resolution}px")
        print(f"    Selected image output extension: {selected_image_extension}")
        print(f"    Selected video output extension: {selected_video_extension}")
        print(f"    Selected video output codec: {selected_video_codec}")
        print(f"    Input resize factor: {int(input_resize_factor * 100)}%")
        print(f"    Output resize factor: {int(output_resize_factor * 100)}%")
        print(f"    Save frames: {selected_keep_frames}")
        print("=" * 50)

        place_stop_button()

        event_stop_upscale_process.clear()
        while True:
            try:
                process_status_q.get_nowait()
            except Empty:
                break

        process_upscale_orchestrator = Process(
            target = upscale_orchestrator,
            args = (
                process_status_q,
                event_stop_upscale_process,
                selected_file_list,
                selected_output_path.get(),
                selected_AI_model,
                selected_AI_multithreading,
                input_resize_factor,
                output_resize_factor,
                selected_gpu,
                tiles_resolution,
                selected_blending_factor,
                selected_keep_frames,
                selected_image_extension,
                selected_video_extension,
                selected_video_codec,
            )
        )
        process_upscale_orchestrator.start()

        Thread(target = check_upscale_steps).start()

# GUI utils function ---------------------------

def user_input_checks() -> bool:
    global selected_file_list
    global selected_AI_model
    global selected_image_extension
    global tiles_resolution
    global input_resize_factor
    global output_resize_factor

    # Selected files
    try: selected_file_list = file_widget.get_selected_file_list()
    except:
        info_message.set("Please select a file")
        return False

    if len(selected_file_list) <= 0:
        info_message.set("Please select a file")
        return False


    # AI model
    if selected_AI_model == MENU_LIST_SEPARATOR[0]:
        info_message.set("Please select the AI model")
        return False


    # Input resize factor
    try: input_resize_factor = int(float(str(selected_input_resize_factor.get())))
    except:
        info_message.set("Input resolution % must be a number")
        return False

    if input_resize_factor > 0: input_resize_factor = input_resize_factor/100
    else:
        info_message.set("Input resolution % must be a value > 0")
        return False


    # Output resize factor
    try: output_resize_factor = int(float(str(selected_output_resize_factor.get())))
    except:
        info_message.set("Output resolution % must be a number")
        return False

    if output_resize_factor > 0: output_resize_factor = output_resize_factor/100
    else:
        info_message.set("Output resolution % must be a value > 0")
        return False


    # VRAM limiter
    try: tiles_resolution = 100 * int(float(str(selected_VRAM_limiter.get())))
    except:
        info_message.set("GPU VRAM value must be a number")
        return False

    if tiles_resolution > 0:
        vram_multiplier = VRAM_model_usage.get(selected_AI_model)

        selected_vram = (vram_multiplier * int(float(str(selected_VRAM_limiter.get()))))
        tiles_resolution = int(selected_vram * 100)
    else:
        info_message.set("GPU VRAM value must be a value > 0")
        return False

    return True

def show_error_message(exception: str) -> None:
    messageBox_title    = "Upscale error"
    messageBox_subtitle = "Please report the error on Github or Telegram"
    messageBox_text     = f"\n {str(exception)} \n"

    MessageBox(
        messageType   = "error",
        title         = messageBox_title,
        subtitle      = messageBox_subtitle,
        default_value = None,
        option_list   = [messageBox_text]
    )

def get_upscale_factor() -> int:
    global selected_AI_model
    if MENU_LIST_SEPARATOR[0] in selected_AI_model: upscale_factor = 0
    elif 'x1' in selected_AI_model: upscale_factor = 1
    elif 'x2' in selected_AI_model: upscale_factor = 2
    elif 'x4' in selected_AI_model: upscale_factor = 4

    return upscale_factor

def inputs_are_busy():
    process = globals().get("process_upscale_orchestrator")
    return process is not None and process.is_alive()


def accept_input_drop(event):
    """The left input pane accepts drops; output/settings controls do not."""
    if inputs_are_busy():
        return False
    x = event.x_root - window.winfo_rootx()
    y = event.y_root - window.winfo_rooty()
    return 0 <= x < window.winfo_width() / 2 and 0 <= y < window.winfo_height()


def select_input_files(paths):
    """Common picker/drop path: append readable media and preserve selection."""
    global file_widget, selected_file_list, input_widgets
    if inputs_are_busy():
        info_message.set("Stop the current job before changing input files")
        return False
    paths = list(paths)
    supported = supported_input_files(paths)
    if not supported:
        info_message.set("No supported input files (folders and other files are ignored)")
        return False
    existing = file_widget.get_selected_file_list() if file_widget is not None else []
    selected = merge_input_files(existing, supported)
    if selected == existing:
        info_message.set("These files are already selected")
        return True

    upscale_factor, input_factor, output_factor = get_values_for_file_widget()
    new_widget = FileWidget(
        master=window, selected_file_list=selected, upscale_factor=upscale_factor,
        input_resize_factor=input_factor, output_resize_factor=output_factor,
        fg_color=background_color, bg_color=background_color,
    )
    readable = new_widget.get_selected_file_list()
    if not any(path in readable for path in supported):
        new_widget.destroy()
        info_message.set("Unable to read the selected images or videos")
        return False
    if file_widget is not None:
        file_widget.destroy()
    for widget in input_widgets:
        widget.destroy()
    input_widgets = ()
    file_widget = new_widget
    selected_file_list = list(readable)
    file_widget.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1.0)
    skipped = len(paths) - len(supported) + len(new_widget.unreadable_files)
    message = f"Ready — {len(readable)} input file(s)"
    if skipped:
        message += f" ({skipped} skipped)"
    info_message.set(message)
    return True


def clear_input_files():
    global file_widget, selected_file_list
    if inputs_are_busy():
        info_message.set("Stop the current job before changing input files")
        return False
    if file_widget is not None:
        file_widget.destroy()
    file_widget = None
    selected_file_list = []
    place_loadFile_section()
    info_message.set("Drop input files here or use SELECT FILES")
    return True


def open_files_action():
    if inputs_are_busy():
        info_message.set("Stop the current job before changing input files")
        return
    paths = filedialog.askopenfilenames()
    if paths:
        select_input_files(paths)

def open_output_path_action():
    asked_selected_output_path = filedialog.askdirectory()
    if asked_selected_output_path == "":
        selected_output_path.set(OUTPUT_PATH_CODED)
    else:
        selected_output_path.set(asked_selected_output_path)




# GUI select from menus functions ---------------------------

def select_AI_from_menu(selected_option: str) -> None:
    global selected_AI_model
    selected_AI_model = selected_option
    update_file_widget(1, 2, 3)

def select_AI_multithreading_from_menu(selected_option: str) -> None:
    global selected_AI_multithreading
    if selected_option == "OFF":
        selected_AI_multithreading = 1
    else:
        selected_AI_multithreading = int(selected_option.split()[0])

def select_blending_from_menu(selected_option: str) -> None:
    global selected_blending_factor

    match selected_option:
        case "OFF": selected_blending_factor = 0
        case "Low":      selected_blending_factor = 0.3
        case "Medium":   selected_blending_factor = 0.5
        case "High":     selected_blending_factor = 0.7

def select_gpu_from_menu(selected_option: str) -> None:
    global selected_gpu
    selected_gpu = selected_option

def select_save_frame_from_menu(selected_option: str):
    global selected_keep_frames
    if   selected_option == "ON":  selected_keep_frames = True
    elif selected_option == "OFF": selected_keep_frames = False

def select_image_extension_from_menu(selected_option: str) -> None:
    global selected_image_extension
    selected_image_extension = selected_option

def select_video_extension_from_menu(selected_option: str) -> None:
    global selected_video_extension
    selected_video_extension = selected_option

def select_video_codec_from_menu(selected_option: str) -> None:
    global selected_video_codec
    selected_video_codec = selected_option




# GUI place functions ---------------------------

def place_github_button():

    def opengithub() -> None: open_browser(githubme, new=1)

    git_button = CTkButton(
        master        = window,
        command       = opengithub,
        image         = logo_git,
        width         = 32,
        height        = 32,
        border_width  = 1,
        fg_color      = "transparent",
        text_color    = text_color,
        border_color  = "#404040",
        anchor        = "center",
        text          = "",
        font          = bold11,
        corner_radius = 1
    )

    git_button.place(relx = column_2 + 0.1, rely = 0.04, anchor = "center")

def place_telegram_button():

    def opentelegram() -> None: open_browser(telegramme, new=1)

    telegram_button = CTkButton(
        master        = window,
        command       = opentelegram,
        image         = logo_telegram,
        width         = 32,
        height        = 32,
        border_width  = 1,
        fg_color      = "transparent",
        text_color    = text_color,
        border_color  = "#404040",
        anchor        = "center",
        text          = "",
        font          = bold11,
        corner_radius = 1
    )

    telegram_button.place(relx = column_2 + 0.055, rely = 0.04, anchor = "center")

def place_loadFile_section():
    global input_widgets
    for widget in input_widgets:
        widget.destroy()
    background = CTkFrame(master = window, fg_color = background_color, corner_radius = 1)

    text_drop = ("DROP IMAGES OR VIDEOS HERE\n\n"
                 "IMAGES • jpg png tif bmp webp heic\n"
                 "VIDEOS • mp4 webm mkv flv gif avi mov mpg qt 3gp")

    input_file_text = CTkLabel(
        master     = window,
        text       = text_drop,
        fg_color   = background_color,
        bg_color   = background_color,
        text_color = text_color,
        width      = 300,
        height     = 150,
        font       = bold13,
        anchor     = "center"
    )

    input_file_button = CTkButton(
        master       = window,
        command      = open_files_action,
        text         = "SELECT FILES",
        width        = 140,
        height       = 30,
        font         = bold12,
        border_width  = 1,
        corner_radius = 1,
        fg_color      = "#282828",
        text_color    = "#E0E0E0",
        border_color  = "#0096FF"
    )

    background.place(relx = 0.0, rely = 0.0, relwidth = 0.5, relheight = 1.0)
    input_file_text.place(relx = 0.25, rely = 0.39, anchor = "center")
    input_file_button.place(relx = 0.25, rely = 0.55, anchor = "center")
    input_widgets = (background, input_file_text, input_file_button)

def place_app_name():
    background = CTkFrame(master = window, fg_color = background_color, corner_radius = 1)
    app_name_label = CTkLabel(
        master     = window,
        text       = app_name + " " + version,
        fg_color   = background_color,
        text_color = app_name_color,
        font       = bold20,
        anchor     = "w"
    )
    background.place(relx = 0.5, rely = 0.0, relwidth = 0.5, relheight = 1.0)
    app_name_label.place(relx = column_1 - 0.05, rely = 0.04, anchor = "center")

def place_AI_menu():

    def open_info_AI_model():
        option_list = [
            "\n IRCNN_Mx1 | IRCNN_Lx1 \n"
            "\n • Simple and lightweight AI models\n"
            " • Year: 2017\n"
            " • Function: Denoising\n",

            "\n RealESR_Gx4 | RealESR_Animex4 \n"
            "\n • Fast and lightweight AI models\n"
            " • Year: 2022\n"
            " • Function: Upscaling\n",

            "\n BSRGANx2 | BSRGANx4 | RealESRGANx4 \n"
            "\n • Complex and heavy AI models\n"
            " • Year: 2020\n"
            " • Function: High-quality upscaling\n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "AI model",
            subtitle      = "This widget allows to choose between different AI models for upscaling",
            default_value = None,
            option_list   = option_list
        )

    widget_row = row1
    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    info_button = create_info_button(open_info_AI_model, "AI model")
    option_menu = create_option_menu(select_AI_from_menu, AI_models_list, default_AI_model)

    info_button.place(relx = column_info1, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_3_5,   rely = widget_row, anchor = "center")

def place_AI_blending_menu():

    def open_info_AI_blending():
        option_list = [
            " Blending combines the upscaled image produced by AI with the original image",

            " \n BLENDING OPTIONS\n" +
            "  • [OFF] No blending is applied\n" +
            "  • [Low] The result favors the upscaled image, with a slight touch of the original\n" +
            "  • [Medium] A balanced blend of the original and upscaled images\n" +
            "  • [High] The result favors the original image, with subtle enhancements from the upscaled version\n",

            " \n NOTES\n" +
            "  • Can enhance the quality of the final result\n" +
            "  • Especially effective when using the tiling/merging function (useful for low VRAM)\n" +
            "  • Particularly helpful at low input resolution percentages (<50%)\n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "AI blending",
            subtitle      = "This widget allows you to choose the blending between the upscaled and original image/frame",
            default_value = None,
            option_list   = option_list
        )

    widget_row = row2

    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    info_button = create_info_button(open_info_AI_blending, "AI blending")
    option_menu = create_option_menu(select_blending_from_menu, blending_list, default_blending)

    info_button.place(relx = column_info1, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_3_5,   rely = widget_row, anchor = "center")

def place_AI_multithreading_menu():

    def open_info_AI_multithreading():
        option_list = [
            " This option can enhance video upscaling performance, especially on powerful GPUs.",

            " \n AI MULTITHREADING OPTIONS\n"
            + "  • OFF - Processes one frame at a time.\n"
            + "  • 2 threads - Processes two frames simultaneously.\n"
            + "  • 4 threads - Processes four frames simultaneously.\n"
            + "  • 6 threads - Processes six frames simultaneously.\n"
            + "  • 8 threads - Processes eight frames simultaneously.\n",

            " \n NOTES\n"
            + "  • Higher thread counts increase CPU, GPU, and RAM usage.\n"
            + "  • The GPU may be heavily stressed, potentially reaching high temperatures.\n"
            + "  • Monitor your system's temperature to prevent overheating.\n"
            + "  • If the chosen thread count exceeds GPU capacity, the app automatically selects an optimal value.\n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "AI multithreading (EXPERIMENTAL)",
            subtitle      = "This widget allows to choose how many video frames are upscaled simultaneously",
            default_value = None,
            option_list   = option_list
        )

    widget_row = row3
    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    info_button = create_info_button(open_info_AI_multithreading, "AI multithreading")
    option_menu = create_option_menu(select_AI_multithreading_from_menu, AI_multithreading_list, default_AI_multithreading)

    info_button.place(relx = column_info1, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_3_5,   rely = widget_row, anchor = "center")

def place_input_output_resolution_textboxs():

    def open_info_input_resolution():
        option_list = [
            " A high value (>50%) will create high quality photos/videos but will be slower",
            " While a low value (<50%) will create good quality photos/videos but will much faster",

            " \n For example, for a 1080p (1920x1080) image/video\n" +
            " • Input scale 25% => input to AI 270p (480x270)\n" +
            " • Input scale 50% => input to AI 540p (960x540)\n" +
            " • Input scale 75% => input to AI 810p (1440x810)\n" +
            " • Input scale 100% => input to AI 1080p (1920x1080) \n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "Input resolution %",
            subtitle      = "This widget allows to choose the resolution input to the AI",
            default_value = None,
            option_list   = option_list
        )

    def open_info_output_resolution():
        option_list = [
            " 100% keeps the exact resolution produced by the AI upscaling",
            " A lower value (<100%) will downscale the AI result to a smaller resolution, saving space and processing time",
            " A higher value (>100%) will further upscale the AI output, increasing size but not adding real details",

            "\n For example, if the AI generates a 4K (3840x2160) image/video\n" +
            " • Output scale 50%  => final output 1920x1080 (downscaled)\n" +
            " • Output scale 100% => final output 3840x2160 (AI native)\n" +
            " • Output scale 200% => final output 7680x4320 (8K, interpolated)\n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "Output resolution %",
            subtitle      = "This widget allows to choose upscaled files resolution",
            default_value = None,
            option_list   = option_list
        )


    widget_row = row4

    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    # Input scale %
    info_button = create_info_button(open_info_input_resolution, "Input scale %")
    option_menu = create_text_box(selected_input_resize_factor, width = little_textbox_width)

    info_button.place(relx = column_info1, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_1_5,   rely = widget_row, anchor = "center")

    # Output scale %
    info_button = create_info_button(open_info_output_resolution, "Output scale %")
    option_menu = create_text_box(selected_output_resize_factor, width = little_textbox_width)

    info_button.place(relx = column_info2, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_3,     rely = widget_row, anchor = "center")

def place_gpu_gpuVRAM_menus():

    def open_info_gpu():
        option_list = [
            "\n It is possible to select up to 4 GPUs for AI processing\n" +
            "  • Auto (the app will select the most powerful GPU)\n" +
            "  • GPU 1 (GPU 0 in Task manager)\n" +
            "  • GPU 2 (GPU 1 in Task manager)\n" +
            "  • GPU 3 (GPU 2 in Task manager)\n" +
            "  • GPU 4 (GPU 3 in Task manager)\n",

            "\n NOTES\n" +
            "  • Keep in mind that the more powerful the chosen gpu is, the faster the upscaling will be\n" +
            "  • For optimal performance, it is essential to regularly update your GPUs drivers\n" +
            "  • Selecting a GPU not present in the PC will cause the app to use the CPU for AI processing\n"
        ]

        MessageBox(
            messageType   = "info",
            title         = "GPU",
            subtitle      = "This widget allows to select the GPU for AI upscale",
            default_value = None,
            option_list   = option_list
        )

    def open_info_vram_limiter():
        option_list = [
            " Make sure to enter the correct value based on the selected GPU's VRAM",
            " Setting a value higher than the available VRAM may cause upscale failure",
            " For integrated GPUs (Intel HD series • Vega 3, 5, 7), select 2 GB to avoid issues",
        ]

        MessageBox(
            messageType   = "info",
            title         = "GPU VRAM (GB)",
            subtitle      = "This widget allows to set a limit on the GPU VRAM memory usage",
            default_value = None,
            option_list   = option_list
        )

    widget_row = row5

    background  = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    # GPU
    info_button = create_info_button(open_info_gpu, "GPU")
    option_menu = create_option_menu(select_gpu_from_menu, gpus_list, default_gpu, width = little_menu_width)

    info_button.place(relx = column_info1,        rely = widget_row, anchor = "center")
    option_menu.place(relx = column_1_4, rely = widget_row,  anchor = "center")

    # GPU VRAM
    info_button = create_info_button(open_info_vram_limiter, "GPU VRAM (GB)")
    option_menu = create_text_box(selected_VRAM_limiter, width = little_textbox_width)

    info_button.place(relx = column_info2, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_3,     rely = widget_row, anchor = "center")

def place_image_video_output_menus():

    def open_info_image_output():
        option_list = [
            " \n PNG\n"
            " • Very good quality\n"
            " • Slow and heavy file\n"
            " • Supports transparent images\n"
            " • Lossless compression (no quality loss)\n"
            " • Ideal for graphics, web images, and screenshots\n",

            " \n JPG\n"
            " • Good quality\n"
            " • Fast and lightweight file\n"
            " • Lossy compression (some quality loss)\n"
            " • Ideal for photos and web images\n"
            " • Does not support transparency\n",

            " \n BMP\n"
            " • Highest quality\n"
            " • Slow and heavy file\n"
            " • Uncompressed format (large file size)\n"
            " • Ideal for raw images and high-detail graphics\n"
            " • Does not support transparency\n",

            " \n TIFF\n"
            " • Highest quality\n"
            " • Very slow and heavy file\n"
            " • Supports both lossless and lossy compression\n"
            " • Often used in professional photography and printing\n"
            " • Supports multiple layers and transparency\n",
        ]


        MessageBox(
            messageType   = "info",
            title         = "Image output",
            subtitle      = "This widget allows to choose the extension of upscaled images",
            default_value = None,
            option_list   = option_list
        )

    def open_info_video_extension():
        option_list = [
            " \n MP4\n"
            " • Most widely supported format\n"
            " • Good quality with efficient compression\n"
            " • Fast and lightweight file\n"
            " • Ideal for streaming and general use\n",

            " \n MKV\n"
            " • High-quality format with multiple audio and subtitle tracks support\n"
            " • Larger file size compared to MP4\n"
            " • Supports almost any codec\n"
            " • Ideal for high-quality videos and archiving\n",

            " \n AVI\n"
            " • Older format with high compatibility\n"
            " • Larger file size due to less efficient compression\n"
            " • Supports multiple codecs but lacks modern features\n"
            " • Ideal for older devices and raw video storage\n",

            " \n MOV\n"
            " • High-quality format developed by Apple\n"
            " • Large file size due to less compression\n"
            " • Best suited for editing and high-quality playback\n"
            " • Compatible mainly with macOS and iOS devices\n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "Video output",
            subtitle      = "This widget allows to choose the extension of the upscaled video",
            default_value = None,
            option_list   = option_list
        )

    widget_row = row6

    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    # Image output
    info_button = create_info_button(open_info_image_output, "Image output")
    option_menu = create_option_menu(select_image_extension_from_menu, image_extension_list, default_image_extension, width = little_menu_width)
    info_button.place(relx = column_info1,        rely = widget_row, anchor = "center")
    option_menu.place(relx = column_1_4, rely = widget_row, anchor = "center")

    # Video output
    info_button = create_info_button(open_info_video_extension, "Video output")
    option_menu = create_option_menu(select_video_extension_from_menu, video_extension_list, default_video_extension, width = little_menu_width)
    info_button.place(relx = column_info2,      rely = widget_row, anchor = "center")
    option_menu.place(relx = column_2_9, rely = widget_row, anchor = "center")

def place_video_codec_keep_frames_menus():

    def open_info_video_codec():
        option_list = [
            " \n SOFTWARE ENCODING (CPU)\n"
            " • x264 | H.264 software encoding\n"
            " • x265 | HEVC (H.265) software encoding\n",

            " \n NVIDIA GPU ENCODING (NVENC - Optimized for NVIDIA GPU)\n"
            " • h264_nvenc | H.264 hardware encoding\n"
            " • hevc_nvenc | HEVC (H.265) hardware encoding\n",

            " \n AMD GPU ENCODING (AMF - Optimized for AMD GPU)\n"
            " • h264_amf | H.264 hardware encoding\n"
            " • hevc_amf | HEVC (H.265) hardware encoding\n",

            " \n INTEL GPU ENCODING (QSV - Optimized for Intel GPU)\n"
            " • h264_qsv | H.264 hardware encoding\n"
            " • hevc_qsv | HEVC (H.265) hardware encoding\n"
        ]


        MessageBox(
            messageType   = "info",
            title         = "Video codec",
            subtitle      = "This widget allows to choose video codec for upscaled video",
            default_value = None,
            option_list   = option_list
        )

    def open_info_keep_frames():
        option_list = [
            "\n ON \n" +
            " Keep the SQLite cache with original/upscaled frames and job settings \n",

            "\n OFF \n" +
            " Delete the cache only after a successful encode; interrupted jobs can resume \n"
        ]

        MessageBox(
            messageType   = "info",
            title         = "Keep video frames",
            subtitle      = "Video frames and resume state are stored in one SQLite file",
            default_value = None,
            option_list   = option_list
        )


    widget_row = row7

    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    # Video codec
    info_button = create_info_button(open_info_video_codec, "Video codec")
    option_menu = create_option_menu(select_video_codec_from_menu, video_codec_list, default_video_codec, width = little_menu_width)
    info_button.place(relx = column_info1,        rely = widget_row, anchor = "center")
    option_menu.place(relx = column_1_4, rely = widget_row, anchor = "center")

    # Keep frames
    info_button = create_info_button(open_info_keep_frames, "Keep frames")
    option_menu = create_option_menu(select_save_frame_from_menu, keep_frames_list, default_keep_frames, width = little_menu_width)
    info_button.place(relx = column_info2,      rely = widget_row, anchor = "center")
    option_menu.place(relx = column_2_9, rely = widget_row, anchor = "center")

def place_output_path_textbox():

    def open_info_output_path():
        option_list = [
              "\n The default path is defined by the input files."
            + "\n For example: selecting a file from the Download folder,"
            + "\n the app will save upscaled files in the Download folder \n",

            " Otherwise it is possible to select the desired path using the SELECT button",
        ]

        MessageBox(
            messageType   = "info",
            title         = "Output path",
            subtitle      = "This widget allows to choose upscaled files path",
            default_value = None,
            option_list   = option_list
        )

    background    = create_option_background()
    info_button   = create_info_button(open_info_output_path, "Output path")
    option_menu   = create_text_box_output_path(selected_output_path)
    active_button = create_active_button(command = open_output_path_action, text = "SELECT", width = 60, height = 25)

    background.place(   relx = 0.75,                 rely = row10, relwidth = 0.48, anchor = "center")
    info_button.place(  relx = column_info1,         rely = row10 - 0.003,           anchor = "center")
    active_button.place(relx = column_info1 + 0.052, rely = row10,                   anchor = "center")
    option_menu.place(  relx = column_2 - 0.008,     rely = row10,                   anchor = "center")

def place_message_label():
    message_label = CTkLabel(
        master        = window,
        textvariable  = info_message,
        height        = 24,
        width         = 247,
        font          = bold11,
        fg_color      = "#ffbf00",
        text_color    = "#000000",
        anchor        = "center",
        corner_radius = 1
    )
    message_label.place(relx = 0.85, rely = 0.9495, anchor = "center")

def place_stop_button():
    stop_button = create_active_button(
        command      = stop_button_command,
        text         = "STOP",
        icon         = stop_icon,
        width        = 140,
        height       = 30,
        border_color = "#EC1D1D"
    )
    stop_button.place(relx = 0.75 - 0.1, rely = 0.95, anchor = "center")

def place_upscale_button():
    upscale_button = create_active_button(
        command = upscale_button_command,
        text    = "UPSCALE",
        icon    = upscale_icon,
        width   = 140,
        height  = 30
    )
    upscale_button.place(relx = 0.75 - 0.1, rely = 0.95, anchor = "center")




# Main functions ---------------------------

def check_directml_support() -> bool:
    providers = onnxruntime_get_available_providers()
    return "DmlExecutionProvider" in providers

def on_app_close() -> None:
    window.grab_release()
    window.destroy()

    global selected_AI_model
    global selected_AI_multithreading
    global selected_gpu
    global selected_blending_factor
    global selected_image_extension
    global selected_video_extension
    global selected_video_codec
    global tiles_resolution
    global input_resize_factor

    AI_model_to_save        = f"{selected_AI_model}"
    gpu_to_save             = selected_gpu
    image_extension_to_save = selected_image_extension
    video_extension_to_save = selected_video_extension
    video_codec_to_save     = selected_video_codec
    blending_to_save        = {0: "OFF", 0.3: "Low", 0.5: "Medium", 0.7: "High"}.get(selected_blending_factor)

    if selected_keep_frames == True:
        keep_frames_to_save = "ON"
    else:
        keep_frames_to_save = "OFF"

    if selected_AI_multithreading == 1:
        AI_multithreading_to_save = "OFF"
    else:
        AI_multithreading_to_save = f"{selected_AI_multithreading} threads"

    user_preference = {
        "default_AI_model":             AI_model_to_save,
        "default_AI_multithreading":    AI_multithreading_to_save,
        "default_gpu":                  gpu_to_save,
        "default_keep_frames":          keep_frames_to_save,
        "default_image_extension":      image_extension_to_save,
        "default_video_extension":      video_extension_to_save,
        "default_video_codec":          video_codec_to_save,
        "default_blending":             blending_to_save,
        "default_output_path":          selected_output_path.get(),
        "default_input_resize_factor":  str(selected_input_resize_factor.get()),
        "default_output_resize_factor": str(selected_output_resize_factor.get()),
        "default_VRAM_limiter":         str(selected_VRAM_limiter.get()),
    }
    user_preference_json = json_dumps(user_preference)
    with open(USER_PREFERENCE_PATH, "w") as preference_file:
        preference_file.write(user_preference_json)

    stop_upscale_process()

class App():
    def __init__(self, window):
        self.toplevel_window = None
        window.protocol("WM_DELETE_WINDOW", on_app_close)

        window.title('')
        window.geometry("1000x675")
        window.resizable(False, False)
        window.iconbitmap(find_by_relative_path("Assets" + os_separator + "logo.ico"))

        place_loadFile_section()

        place_app_name()
        place_output_path_textbox()
        place_github_button()
        place_telegram_button()

        place_AI_menu()
        place_AI_blending_menu()
        place_AI_multithreading_menu()
        place_input_output_resolution_textboxs()
        place_gpu_gpuVRAM_menus()
        place_image_video_output_menus()
        place_video_codec_keep_frames_menus()

        place_message_label()
        place_upscale_button()

def main():
    global default_AI_model, default_AI_multithreading, default_gpu, default_keep_frames
    global default_image_extension, default_video_extension, default_video_codec
    global default_blending, default_output_path, default_VRAM_limiter
    global default_input_resize_factor, default_output_resize_factor
    global process_status_q, event_stop_upscale_process, window, info_message
    global selected_output_path, selected_input_resize_factor, selected_output_resize_factor
    global selected_VRAM_limiter
    global bold8, bold9, bold10, bold11, bold12, bold13, bold14, bold16
    global bold17, bold18, bold19, bold20, bold21, bold22, bold23, bold24
    global logo_git, logo_telegram, stop_icon, upscale_icon, clear_icon, info_icon
    global file_widget, input_widgets

    if os_path_exists(FFMPEG_EXE_PATH):
        print(f"[{app_name}] ffmpeg.exe found")
    else:
        print(f"[{app_name}] ffmpeg.exe not found, please install ffmpeg.exe following the guide")

    if os_path_exists(USER_PREFERENCE_PATH):
        print(f"[{app_name}] Preference file exist")
        with open(USER_PREFERENCE_PATH, "r") as json_file:
            json_data = json_load(json_file)
            default_AI_model             = json_data.get("default_AI_model",             AI_models_list[0])
            default_AI_multithreading    = json_data.get("default_AI_multithreading",    AI_multithreading_list[0])
            default_gpu                  = json_data.get("default_gpu",                  gpus_list[0])
            default_keep_frames          = json_data.get("default_keep_frames",          keep_frames_list[1])
            default_image_extension      = json_data.get("default_image_extension",      image_extension_list[0])
            default_video_extension      = json_data.get("default_video_extension",      video_extension_list[0])
            default_video_codec          = json_data.get("default_video_codec",          video_codec_list[0])
            default_blending             = json_data.get("default_blending",             blending_list[1])
            default_output_path          = json_data.get("default_output_path",          OUTPUT_PATH_CODED)
            default_input_resize_factor  = json_data.get("default_input_resize_factor",  str(50))
            default_output_resize_factor = json_data.get("default_output_resize_factor", str(100))
            default_VRAM_limiter         = json_data.get("default_VRAM_limiter",         str(4))
    else:
        print(f"[{app_name}] Preference file does not exist, using default coded value")
        default_AI_model             = AI_models_list[0]
        default_AI_multithreading    = AI_multithreading_list[0]
        default_gpu                  = gpus_list[0]
        default_keep_frames          = keep_frames_list[1]
        default_image_extension      = image_extension_list[0]
        default_video_extension      = video_extension_list[0]
        default_video_codec          = video_codec_list[0]
        default_blending             = blending_list[1]
        default_output_path          = OUTPUT_PATH_CODED
        default_input_resize_factor  = str(50)
        default_output_resize_factor = str(100)
        default_VRAM_limiter         = str(4)

    multiprocessing_freeze_support()
    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    # Multiprocessing utilities
    multiprocessing_manager    = multiprocessing_Manager()
    process_status_q           = multiprocessing_manager.Queue(maxsize=1)
    event_stop_upscale_process = multiprocessing_Event()

    window = FileDropWindow()
    file_widget = None
    input_widgets = ()
    info_message                  = StringVar()
    selected_output_path          = StringVar()
    selected_input_resize_factor  = StringVar()
    selected_output_resize_factor = StringVar()
    selected_VRAM_limiter         = StringVar()

    global selected_file_list
    global selected_AI_model
    global selected_blending_factor
    global selected_AI_multithreading
    global selected_gpu
    global selected_keep_frames
    global selected_image_extension
    global selected_video_extension
    global selected_video_codec
    global tiles_resolution
    global input_resize_factor

    selected_file_list = []

    selected_AI_model        = default_AI_model
    selected_gpu             = default_gpu
    selected_image_extension = default_image_extension
    selected_video_extension = default_video_extension
    selected_video_codec     = default_video_codec

    if default_AI_multithreading == "OFF":
        selected_AI_multithreading = 1
    else:
        selected_AI_multithreading = int(default_AI_multithreading.split()[0])

    if default_keep_frames == "ON":
        selected_keep_frames = True
    else:
        selected_keep_frames = False

    selected_blending_factor = {"OFF": 0, "Low": 0.3, "Medium": 0.5, "High": 0.7}.get(default_blending)

    selected_input_resize_factor.set(default_input_resize_factor)
    selected_output_resize_factor.set(default_output_resize_factor)
    selected_VRAM_limiter.set(default_VRAM_limiter)
    selected_output_path.set(default_output_path)

    info_message.set("Hi :)")
    selected_input_resize_factor.trace_add('write', update_file_widget)
    selected_output_resize_factor.trace_add('write', update_file_widget)

    font   = "Segoe UI"
    bold8  = CTkFont(family = font, size = 8, weight = "bold")
    bold9  = CTkFont(family = font, size = 9, weight = "bold")
    bold10 = CTkFont(family = font, size = 10, weight = "bold")
    bold11 = CTkFont(family = font, size = 11, weight = "bold")
    bold12 = CTkFont(family = font, size = 12, weight = "bold")
    bold13 = CTkFont(family = font, size = 13, weight = "bold")
    bold14 = CTkFont(family = font, size = 14, weight = "bold")
    bold16 = CTkFont(family = font, size = 16, weight = "bold")
    bold17 = CTkFont(family = font, size = 17, weight = "bold")
    bold18 = CTkFont(family = font, size = 18, weight = "bold")
    bold19 = CTkFont(family = font, size = 19, weight = "bold")
    bold20 = CTkFont(family = font, size = 20, weight = "bold")
    bold21 = CTkFont(family = font, size = 21, weight = "bold")
    bold22 = CTkFont(family = font, size = 22, weight = "bold")
    bold23 = CTkFont(family = font, size = 23, weight = "bold")
    bold24 = CTkFont(family = font, size = 24, weight = "bold")

    # Images
    logo_git      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}github_logo.png")),    size=(22, 22))
    logo_telegram = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}telegram_logo.png")),  size=(18, 18))
    stop_icon     = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}stop_icon.png")),      size=(15, 15))
    upscale_icon  = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}upscale_icon.png")),   size=(15, 15))
    clear_icon    = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}clear_icon.png")),     size=(15, 15))
    info_icon     = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}info_icon.png")),      size=(18, 18))

    app = App(window)
    window.file_drop_target = FileDropTarget(
        window, select_input_files, accepts=accept_input_drop,
        on_invalid=lambda: info_message.set("Unable to read the dropped file paths"),
    )
    window.update()
    window.mainloop()
    multiprocessing_manager.shutdown()
