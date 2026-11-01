"""Supported input extensions shared by the GUI and batch backend."""

IMAGE_EXTENSIONS = {".heic", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {
    ".mp4", ".webm", ".mkv", ".flv", ".gif", ".m4v", ".avi", ".mov",
    ".qt", ".3gp", ".mpg", ".mpeg", ".vob", ".wmv",
}
INPUT_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
