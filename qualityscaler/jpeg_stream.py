"""Read concatenated JPEG marker streams, such as FFmpeg's image2pipe output.

The parser validates marker framing, segment lengths, and truncation, not image
dimensions, coding tables, or the compressed pixels themselves. Segment payloads
are opaque: embedded SOI/EOI bytes in APP/COM/table metadata are not boundaries.
Only the current frame and a bounded input chunk are buffered.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import BinaryIO


class _Reader:
    def __init__(self, stream: BinaryIO, chunk_size: int) -> None:
        self.stream = stream
        self.chunk_size = chunk_size
        self.buffer = b""
        self.position = 0

    def fill(self) -> bool:
        if self.position < len(self.buffer):
            return True
        data = self.stream.read(self.chunk_size)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("JPEG stream must be binary and read() must return bytes")
        if len(data) > self.chunk_size:
            raise ValueError("JPEG stream returned more bytes than requested")
        self.buffer = bytes(data)
        self.position = 0
        return bool(self.buffer)

    def byte(self, frame: bytearray | None = None) -> int | None:
        if not self.fill():
            return None
        value = self.buffer[self.position]
        self.position += 1
        if frame is not None:
            frame.append(value)
        return value

    def required_byte(self, frame: bytearray, description: str) -> int:
        value = self.byte(frame)
        if value is None:
            raise ValueError(f"truncated JPEG {description}")
        return value

    def payload(self, size: int, frame: bytearray) -> None:
        while size:
            if not self.fill():
                raise ValueError("truncated JPEG segment payload")
            end = min(len(self.buffer), self.position + size)
            frame.extend(self.buffer[self.position : end])
            size -= end - self.position
            self.position = end

    def scan_to_marker(self, frame: bytearray) -> None:
        """Append entropy bytes up to, but not including, the next FF byte."""
        while True:
            if not self.fill():
                raise ValueError("truncated JPEG scan (missing EOI marker)")
            end = self.buffer.find(b"\xff", self.position)
            if end != -1:
                frame.extend(self.buffer[self.position : end])
                self.position = end
                return
            frame.extend(self.buffer[self.position :])
            self.position = len(self.buffer)


def iter_jpeg_frames(
    stream: BinaryIO, chunk_size: int = 16384
) -> Iterator[bytes]:
    """Yield exact JPEG bytes from a binary stream of concatenated frames.

    Short reads and markers split across reads are supported, including
    one-byte reads. Stuffed bytes, restart markers, marker fill bytes, and
    progressive/multiple scans are preserved. Empty input yields nothing.
    Malformed/truncated input and any garbage between/after frames raise
    ``ValueError``; a nonbinary stream raises ``TypeError``.

    The caller owns the stream; this generator never closes it. A valid frame
    can be yielded before an error is found in a later frame or trailing data.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    reader = _Reader(stream, chunk_size)
    while True:
        first = reader.byte()
        if first is None:
            return
        if first != 0xFF:
            raise ValueError("expected JPEG SOI marker; found trailing/leading garbage")
        second = reader.byte()
        if second is None:
            raise ValueError("truncated JPEG SOI marker")
        if second != 0xD8:
            raise ValueError("expected JPEG SOI marker")
        frame = bytearray(b"\xff\xd8")
        in_scan = False

        while True:
            if in_scan:
                reader.scan_to_marker(frame)
            prefix = reader.required_byte(frame, "marker")
            if prefix != 0xFF:
                raise ValueError("expected JPEG marker prefix")
            marker = reader.required_byte(frame, "marker")
            while marker == 0xFF:
                marker = reader.required_byte(frame, "marker")

            if marker == 0x00:
                if not in_scan:
                    raise ValueError("stuffed JPEG byte outside a scan")
                continue
            if 0xD0 <= marker <= 0xD7:
                if not in_scan:
                    raise ValueError("JPEG restart marker outside a scan")
                continue
            if marker == 0x01:  # TEM is a standalone marker, without a length.
                continue
            if marker == 0xD8:
                raise ValueError("unexpected JPEG SOI marker inside a frame")
            if marker == 0xD9:
                result = bytes(frame)
                del frame
                yield result
                del result
                break
            if marker < 0xC0:
                raise ValueError(f"invalid JPEG marker 0xFF{marker:02X}")

            high = reader.required_byte(frame, "segment length")
            low = reader.required_byte(frame, "segment length")
            length = (high << 8) | low
            if length < 2:
                raise ValueError("JPEG segment length must be at least two")
            reader.payload(length - 2, frame)

            # Most non-entropy markers terminate a scan. DNL is special: it
            # can supply a line count in the middle of an entropy-coded scan.
            in_scan = marker == 0xDA or (in_scan and marker == 0xDC)
