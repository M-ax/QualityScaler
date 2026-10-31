"""Batch and still-image orchestration, independent of the GUI toolkit."""

from .ai import AI_upscale
from .ffmpeg import UpscaleCancelled
from .formats import VIDEO_EXTENSIONS
from .media import blend_images_and_save, copy_file_metadata, image_read, image_write
from .paths import prepare_output_image_filename
from .status import write_process_status
from .video import upscale_video


def check_if_file_is_video(path):
    from pathlib import Path
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def upscale_image(process_status_q, image_path, file_number, selected_output_path,
                  AI_instance, selected_AI_model, selected_image_extension,
                  input_resize_factor, output_resize_factor, selected_blending_factor):
    original = image_read(image_path)
    output = prepare_output_image_filename(
        image_path, selected_output_path, selected_AI_model, input_resize_factor,
        output_resize_factor, selected_image_extension, selected_blending_factor,
    )
    write_process_status(process_status_q, f"{file_number}. Upscaling image")
    upscaled = AI_instance.AI_orchestration(original)
    if selected_blending_factor:
        blend_images_and_save(
            output, original, upscaled, selected_blending_factor, selected_image_extension
        )
    else:
        image_write(output, upscaled, selected_image_extension)
    copy_file_metadata(image_path, output)


def upscale_orchestrator(
    process_status_q, event_stop_upscale_process, selected_file_list,
    selected_output_path, selected_AI_model, selected_AI_multithreading,
    input_resize_factor, output_resize_factor, selected_gpu, tiles_resolution,
    selected_blending_factor, selected_keep_frames, selected_image_extension,
    selected_video_extension, selected_video_codec,
):
    try:
        for file_number, path in enumerate(selected_file_list, 1):
            if event_stop_upscale_process.is_set():
                raise UpscaleCancelled()
            if check_if_file_is_video(path):
                upscale_video(
                    process_status_q, event_stop_upscale_process, path, file_number,
                    selected_output_path, selected_AI_model, selected_blending_factor,
                    selected_AI_multithreading, selected_gpu, input_resize_factor,
                    output_resize_factor, tiles_resolution, selected_video_extension,
                    selected_video_codec, selected_keep_frames,
                )
            else:
                upscale_image(
                    process_status_q, path, file_number, selected_output_path,
                    AI_upscale(selected_AI_model, selected_gpu, input_resize_factor,
                               output_resize_factor, tiles_resolution),
                    selected_AI_model, selected_image_extension, input_resize_factor,
                    output_resize_factor, selected_blending_factor,
                )
        write_process_status(
            process_status_q, "Stop" if event_stop_upscale_process.is_set() else "Completed"
        )
    except UpscaleCancelled:
        write_process_status(process_status_q, "Stop")
    except Exception as error:
        write_process_status(process_status_q, f"Error {error}")
