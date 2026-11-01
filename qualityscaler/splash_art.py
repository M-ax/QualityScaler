"""Build-time JPEG artifacts for the native splash; no runtime JPEG codec."""

from base64 import b64encode
from io import BytesIO

from PIL import Image, ImageChops


# (load fraction, JPEG quality, lossy generations). Keep most of loading rough;
# recover quickly near readiness, then let Tcl restore the original at 100%.
JPEG_STAGES = (
    (0.00, 1, 8),
    (0.75, 1, 3),
    (0.80, 1, 1),
    (0.84, 3, 1),
    (0.87, 6, 1),
    (0.89, 12, 1),
    (0.91, 30, 1),
    (0.93, 60, 1),
    (0.94, 92, 1),
)


def jpeg_artifact_script(image_file, max_img_size: tuple[int, int]) -> str:
    """Embed full-resolution JPEG round trips as PNG photos for stock Tcl/Tk."""
    with Image.open(image_file) as source:
        image = source.convert("RGB")
    scale = min(1.0, max_img_size[0] / image.width, max_img_size[1] / image.height)
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)))

    thresholds = " ".join(str(threshold) for threshold, _, _ in JPEG_STAGES)
    script = [f"set qs_quality_thresholds {{{thresholds}}}", "set qs_quality_images [list]"]
    for _, quality, generations in JPEG_STAGES:
        # Keep the displayed resolution fixed. Actual JPEG quantization and
        # chroma subsampling supply ringing, block boundaries and color bleed.
        frame = image
        for generation in range(generations):
            # Move the codec's block grid between generations to accumulate
            # damage instead of requantizing identical blocks. Undo the shift
            # after decoding; this never shrinks or enlarges the image.
            shift = (generation * 3) % 8
            shifted = ImageChops.offset(frame, shift, shift)
            jpeg = BytesIO()
            shifted.save(jpeg, format="JPEG", quality=quality, subsampling=2)
            jpeg.seek(0)
            with Image.open(jpeg) as compressed:
                frame = ImageChops.offset(compressed, -shift, -shift)
        png = BytesIO()
        frame.save(png, format="PNG")
        # Minimal native Tk reads PNG, not JPEG. Encoding the decoded pixels as
        # PNG preserves the JPEG artifacts without adding a runtime dependency.
        data = b64encode(png.getvalue()).decode("ascii")
        script.append(f"lappend qs_quality_images [image create photo -data {{{data}}}]")
    return "\n".join(script) + "\n"
