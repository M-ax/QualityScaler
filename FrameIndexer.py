from typing_extensions import *


class FrameIndexer:
    def __init__(
            self,
            in_stream : IO[AnyStr],
            file_name : str,
            page_size: int | None = None
    ):
        self.in_stream = in_stream
        self.file = open(file_name, "a+b")
        self.file.truncate()
        if page_size: self.PAGE_SIZE = page_size

    PAGE_SIZE = 16384

    # JPEG magic frame start bytes
    START_BYTES = bytes([0xFF, 0xD8])

    # JPEG magic frame end bytes
    END_BYTES = bytes([0xFF, 0xD9])

    cur_page : bytes
    pages_consumed = 0
    check_split_marker = False

    # Captured jpeg image indices
    jpg_start = -1
    jpg_end = -1
    jpg_segments = []

    def read_to_end(self):
        try:
            while True:
                self.read_page()
                if len(self.cur_page) == 0: break

                self.file.write(self.cur_page)
                self.parse_page()
                self.pages_consumed += 1
            self.file.seek(0)
        except Exception as e:
            self.file.close()

    def parse_page(self):
        cursor = 0
        while cursor < self.PAGE_SIZE:
            if cursor < 0: cursor = 0
            if self.jpg_start > -1:
                end = -1
                if self.check_split_marker and self.cur_page[0] == self.END_BYTES[1]:
                    # Sneaky end marker
                    # Continue while using -1 as the end, which is correct
                    # as the start of this end marker was the last byte of the last page
                    self.check_split_marker = False
                    cursor = 1
                else:
                    # Look for an end first on this new page
                    end = self.cur_page.find(self.END_BYTES, cursor)
                    if end <= -1: break # No end found, give up on this page
                    cursor = end + 1

                # segment should include the end bytes too
                end += len(self.END_BYTES) - 1

                # We have a start and end, add a segment
                self.jpg_end = end + (self.pages_consumed * self.PAGE_SIZE)
                self.add_segment()
                continue

            start = -1
            if self.check_split_marker and self.cur_page[0] == self.START_BYTES[1]:
                # Sneaky start marker
                self.check_split_marker = False
                cursor = 1
            else:
                start = self.cur_page.find(self.START_BYTES, cursor)
                if start <= -1: break # No start index found, give up on this page
                cursor = start

            # Write down this start index, come back around
            self.jpg_start = start + (self.pages_consumed * self.PAGE_SIZE)

        # Catch any sneaky little shits with the marker bytes split across two pages
        self.check_split_marker = self.cur_page[-1] == self.START_BYTES[0]

    def add_segment(self):
        self.jpg_segments.append((self.jpg_start, self.jpg_end))
        self.jpg_end = self.jpg_start = -1

    def read_page(self):
        self.cur_page = self.in_stream.read(self.PAGE_SIZE)

    def get_frame(self, index : int):
        (start, end) = self.jpg_segments[index]
        self.file.seek(start)
        frame_data = self.file.read(end - start + 1)
        return frame_data
        # assert frame_data[0] == 0xFF
        # assert frame_data[-1] == 0xD9