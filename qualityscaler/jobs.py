"""Batch and still-image orchestration, independent of the GUI toolkit."""

from .ai import AI_upscale
from .ffmpeg import UpscaleCancelled
from .formats import VIDEO_EXTENSIONS
from .media import image_read, image_write, resize_with_output_factor, sharpen_and_save
from .paths import prepare_output_image_filename
from .status import write_process_status
from .video import upscale_video


def check_if_file_is_video(path):
    from pathlib import Path
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def upscale_image(process_status_q, image_path, file_number, selected_output_path,
                  AI_instance, selected_AI_model, selected_image_extension,
                  input_resize_factor, output_resize_factor, selected_sharpening_amount):
    original = image_read(image_path)
    output = prepare_output_image_filename(
        image_path, selected_output_path, selected_AI_model, input_resize_factor,
        output_resize_factor, selected_image_extension, selected_sharpening_amount,
    )
    write_process_status(process_status_q, f"{file_number}. Upscaling image")
    upscaled = AI_instance.AI_orchestration(original)
    upscaled = resize_with_output_factor(upscaled, output_resize_factor)
    if selected_sharpening_amount:
        sharpen_and_save(output, upscaled, selected_sharpening_amount, selected_image_extension)
    else:
        image_write(output, upscaled, selected_image_extension)


def upscale_orchestrator(
    process_status_q, event_stop_upscale_process, selected_file_list,
    selected_output_path, selected_AI_model, selected_AI_multithreading,
    input_resize_factor, output_resize_factor, selected_gpu, tiles_resolution,
    selected_sharpening_amount, selected_keep_frames, selected_image_extension,
    selected_video_extension, selected_video_codec,
):
    try:
        from pathlib import Path
        image_engine = None
        for file_number, path in enumerate(selected_file_list, 1):
            if event_stop_upscale_process.is_set():
                raise UpscaleCancelled()
            if not Path(path).is_file():
                raise FileNotFoundError(f"File not found (moved, renamed or deleted?): {path}")
            if check_if_file_is_video(path):
                upscale_video(
                    process_status_q, event_stop_upscale_process, path, file_number,
                    selected_output_path, selected_AI_model, selected_sharpening_amount,
                    selected_AI_multithreading, selected_gpu, input_resize_factor,
                    output_resize_factor, tiles_resolution, selected_video_extension,
                    selected_video_codec, selected_keep_frames,
                )
            else:
                if image_engine is None:
                    image_engine = AI_upscale(
                        selected_AI_model, selected_gpu, input_resize_factor, tiles_resolution
                    )
                upscale_image(
                    process_status_q, path, file_number, selected_output_path,
                    image_engine,
                    selected_AI_model, selected_image_extension, input_resize_factor,
                    output_resize_factor, selected_sharpening_amount,
                )
        write_process_status(
            process_status_q, "Stop" if event_stop_upscale_process.is_set() else "Completed"
        )
    except UpscaleCancelled:
        write_process_status(process_status_q, "Stop")
    except Exception as error:
        write_process_status(process_status_q, f"Error {error}")
