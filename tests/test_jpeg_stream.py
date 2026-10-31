import base64
import io
import unittest

from qualityscaler.jpeg_stream import iter_jpeg_frames


def segment(marker, payload=b""):
    return b"\xff" + bytes([marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def synthetic_jpeg(entropy=b"\x12\x34", metadata=b"metadata", progressive=False):
    # Marker-level fixture: one-component SOF/SOS, without decoding pixels.
    sof = 0xC2 if progressive else 0xC0
    header = (
        b"\xff\xd8"
        + segment(0xE1, metadata)
        + segment(sof, b"\x08\x00\x01\x00\x01\x01\x01\x11\x00")
    )
    scan = segment(0xDA, b"\x01\x01\x00\x00\x3f\x00") + entropy
    if progressive:
        scan += segment(0xC4, b"table\xff\xd9") + segment(
            0xDA, b"\x01\x01\x00\x01\x3f\x00"
        ) + b"\x56\xff\x00\x78\xff\xd3"
    return header + scan + b"\xff\xd9"


# A real 1x1 baseline JFIF JPEG from FFmpeg, kept inline so tests need no image
# libraries. Its entropy data was also verified with FFmpeg's JPEG decoder.
REAL_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzYyLjExLjEwMAD/2wBDAAgEBAQEBAUFBQUF"
    "BQYGBgYGBgYGBgYGBgYHBwcICAgHBwcGBgcHCAgICAkJCQgICAgJCQoKCgwMCwsODg4RERT/"
    "xABNAAEBAAAAAAAAAAAAAAAAAAAABgEBAQEAAAAAAAAAAAAAAAAAAAYHEAEAAAAAAAAAAAAA"
    "AAAAAAAAEQEAAAAAAAAAAAAAAAAAAAAA/8AAEQgAAQABAwESAAISAAMSAP/aAAwDAQACEQMR"
    "AD8AixJjfx//2Q=="
)


class ShortReadStream:
    def __init__(self, data, limits=(1, 3, 2, 7)):
        self.data = data
        self.position = 0
        self.limits = limits
        self.calls = 0
        self.maximum_request = 0

    def read(self, size):
        self.maximum_request = max(self.maximum_request, size)
        limit = self.limits[self.calls % len(self.limits)]
        self.calls += 1
        end = min(len(self.data), self.position + min(size, limit))
        result = self.data[self.position : end]
        self.position = end
        return result


class JpegStreamTests(unittest.TestCase):
    def test_empty_binary_input(self):
        self.assertEqual(list(iter_jpeg_frames(io.BytesIO())), [])

    def test_real_jpeg_and_multiple_frames(self):
        frames = [REAL_JPEG, synthetic_jpeg(), REAL_JPEG]
        stream = io.BytesIO(b"".join(frames))
        self.assertEqual(list(iter_jpeg_frames(stream)), frames)
        self.assertFalse(stream.closed)

    def test_metadata_contains_embedded_soi_eoi_and_other_markers(self):
        metadata = b"EXIF\xff\xd8fake\xff\xd9\xff\xda\x00\x00\xff\x00"
        frame = synthetic_jpeg(metadata=metadata)
        frames = [frame, synthetic_jpeg(metadata=b"second")]
        for chunk_size in (1, 2, 3, 7, 16, 16384):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(
                    list(iter_jpeg_frames(io.BytesIO(b"".join(frames)), chunk_size)),
                    frames,
                )

    def test_stuffed_bytes_restarts_tem_fill_and_progressive_scans(self):
        entropy = (
            b"\x10\xff\x00\xd8\xff\x00\xd9"
            + b"".join(b"\xff" + bytes([marker]) for marker in range(0xD0, 0xD8))
            + b"\xff\x01\x20\xff\xff\xd0\x30"
        )
        frame = synthetic_jpeg(entropy=entropy, progressive=True)
        frame = frame.replace(b"\xff\xe1", b"\xff\xff\xe1", 1)
        frame = frame[:-2] + b"\xff\xff\xd9"
        for chunk_size in range(1, 18):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(
                    list(iter_jpeg_frames(io.BytesIO(frame + frame), chunk_size)),
                    [frame, frame],
                )

    def test_dnl_marker_can_continue_entropy_scan(self):
        frame = synthetic_jpeg(
            entropy=b"\x12" + segment(0xDC, b"\x00\x01") + b"\x34\xff\x00\x56"
        )
        self.assertEqual(list(iter_jpeg_frames(io.BytesIO(frame), 1)), [frame])

    def test_short_reads_and_one_byte_reads(self):
        frames = [REAL_JPEG, synthetic_jpeg(progressive=True), REAL_JPEG]
        for limits in ((1,), (1, 3, 2, 7), (13, 2, 1, 32)):
            with self.subTest(limits=limits):
                stream = ShortReadStream(b"".join(frames), limits)
                self.assertEqual(list(iter_jpeg_frames(stream, 19)), frames)
                self.assertEqual(stream.maximum_request, 19)

    def test_every_possible_two_chunk_marker_boundary(self):
        frame = synthetic_jpeg(
            metadata=b"embedded\xff\xd9\xff\xd8", progressive=True
        )
        for split in range(1, len(frame)):
            with self.subTest(split=split):
                stream = ShortReadStream(frame, (split, len(frame)))
                self.assertEqual(list(iter_jpeg_frames(stream, len(frame))), [frame])

    def test_large_payload_and_scan_remain_chunked(self):
        frame = synthetic_jpeg(
            metadata=b"x" * 65533,
            entropy=b"\x12" * 100000 + b"\xff\x00" + b"\x34" * 100000,
        )
        stream = ShortReadStream(frame, (1000000,))
        self.assertEqual(list(iter_jpeg_frames(stream, 1024)), [frame])
        self.assertEqual(stream.maximum_request, 1024)

    def test_all_truncated_prefixes_fail_except_empty(self):
        for frame in (REAL_JPEG, synthetic_jpeg(progressive=True)):
            for size in range(1, len(frame)):
                with self.subTest(size=size, length=len(frame)):
                    with self.assertRaises(ValueError):
                        list(iter_jpeg_frames(io.BytesIO(frame[:size]), 3))

    def test_invalid_markers_lengths_and_missing_eoi(self):
        cases = {
            "not JPEG": b"abc",
            "wrong initial marker": b"\xff\xe0\x00\x02",
            "stuffed outside scan": b"\xff\xd8\xff\x00\xff\xd9",
            "restart outside scan": b"\xff\xd8\xff\xd0\xff\xd9",
            "nested SOI": b"\xff\xd8\xff\xd8\xff\xd9",
            "reserved marker": b"\xff\xd8\xff\x02\xff\xd9",
            "missing prefix": b"\xff\xd8garbage\xff\xd9",
            "zero length": b"\xff\xd8\xff\xe1\x00\x00\xff\xd9",
            "one length": b"\xff\xd8\xff\xe1\x00\x01\xff\xd9",
            "oversized payload": b"\xff\xd8\xff\xe1\xff\xffabc\xff\xd9",
            "no scan EOI": synthetic_jpeg()[:-2] + b"remaining entropy",
            "scan nested SOI": synthetic_jpeg(entropy=b"\xff\xd8"),
            "scan reserved marker": synthetic_jpeg(entropy=b"\xff\x03"),
        }
        for name, data in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                list(iter_jpeg_frames(io.BytesIO(data), 2))

    def test_rejects_garbage_between_or_after_frames(self):
        frame = synthetic_jpeg()
        for garbage in (b"\x00", b" ", b"trailing", b"\xff", b"\xff\xd9"):
            for suffix in (b"", frame):
                with self.subTest(garbage=garbage, suffix=bool(suffix)):
                    iterator = iter_jpeg_frames(io.BytesIO(frame + garbage + suffix), 1)
                    self.assertEqual(next(iterator), frame)
                    with self.assertRaises(ValueError):
                        list(iterator)

    def test_binary_only_and_invalid_read_results(self):
        for stream in (io.StringIO(""), io.StringIO("text")):
            with self.assertRaisesRegex(TypeError, "binary"):
                list(iter_jpeg_frames(stream))

        class NoneStream:
            def read(self, size):
                return None

        class OversizedStream:
            def read(self, size):
                return b"x" * (size + 1)

        with self.assertRaises(TypeError):
            list(iter_jpeg_frames(NoneStream()))
        with self.assertRaises(ValueError):
            list(iter_jpeg_frames(OversizedStream(), 1))

    def test_invalid_chunk_sizes(self):
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                list(iter_jpeg_frames(io.BytesIO(), size))
        for size in (None, 1.5, "1", True):
            with self.subTest(size=size), self.assertRaises(TypeError):
                list(iter_jpeg_frames(io.BytesIO(), size))


if __name__ == "__main__":
    unittest.main()
