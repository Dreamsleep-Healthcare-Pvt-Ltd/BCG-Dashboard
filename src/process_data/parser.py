import struct
import logging
from typing import Iterator, Tuple, List, Optional
import plotly.graph_objects as go


FRAME_SIZE  = 813
HEADER_SIZE = 10
MAGIC_HEADER = b'U\xaa'
MAGIC_FOOTER = b'\xaaU'


logger = logging.getLogger(__name__)


class Reader:
    """ Parses binary files containing sensor data.
        Each frame is validated for header, footer, and checksum.
        Provides methods to read BCG data from the frames.
        Usage:
            reader = Reader("datafile.bin")
            channel_1, channel_2, gain_list, epoch_time = reader.read_channels()
        
       """

    def __init__(self, filename: str) -> None:
        self.filename = filename

    def _extract_frames(self, data: bytes, start_magic: bytes, end_magic: bytes) -> Iterator[bytes]:
        """
        Action:
            Extract frames from raw data based on start and end magic bytes.
        Input:
            data: Raw bytes data from the file.
            start_magic: Byte sequence indicating the start of a frame.
            end_magic: Byte sequence indicating the end of a frame.
        Yields:
            Complete frames as bytes.
        """
        pos = 0
        total_len = len(data)
        while pos < total_len:
            start = data.find(start_magic, pos)
            if start == -1:
                return
            end = data.find(end_magic, start + len(start_magic))
            if end == -1:
                return
            yield data[start:end + len(end_magic)]
            pos = end + len(end_magic)

    def _calculate_checksum(self, buf: bytes) -> int:
        """
        Actction:
            Calculate checksum by XORing all bytes in the buffer.
        Input:
            buf: Byte buffer to calculate checksum for.
        Returns:
            Calculated checksum as an integer.
        """
        c = 0
        for b in buf:
            c ^= b
        return c

    def frames(self) -> Iterator[Tuple[int, int, int, bytes, int, int, int]]:
        """
        Action:
            Yield parsed frames as tuples:
        Input:
            None
        Yields:
            Tuples of sensor_type, epoch_time, frame_counter, payload, metadata_bytes, sensor_status, checksum
        """
        with open(self.filename, "rb") as f:
            data = f.read()
        for frame in self._extract_frames(data, MAGIC_HEADER, MAGIC_FOOTER):
            print(len(frame))
            if len(frame) != FRAME_SIZE:
                logger.debug("Skipping frame with unexpected length %d", len(frame))
                continue

            # verify header (first 2 bytes)
            if struct.unpack_from("<H", frame, 0)[0] != struct.unpack_from("<H", MAGIC_HEADER, 0)[0]:
                logger.debug("Invalid header signature, skipping")
                continue

            sensor_type = frame[2]
            epoch_time = struct.unpack_from(">I", frame, 3)[0]
            frame_counter = struct.unpack_from("<H", frame, 7)[0]
            metadata_bytes = frame[8]
            sensor_status = frame[9]

            payload = frame[HEADER_SIZE:]  # includes checksum and footer at tail
            if payload[-2:] != MAGIC_FOOTER:
                logger.debug("Invalid footer, skipping frame %d", frame_counter)
                continue

            org_checksum = payload[-3]
            checksum = self._calculate_checksum(frame[2:-3])

            if checksum != org_checksum:
                logger.debug("Checksum mismatch on frame %d: %d != %d", frame_counter, checksum, org_checksum)
                continue

            yield sensor_type, epoch_time, frame_counter, payload, metadata_bytes, sensor_status, checksum

    def read_channels(self) -> Tuple[List[int], List[int], List[int], Optional[int]]:
        """
        Action:
            Read and parse BCG channel data from the binary file.
        Input:
            None
        Output:
            channel_1: List of samples from channel 1.
            channel_2: List of samples from channel 2.
            gain_list: List of gain values from metadata.
            last_epoch: Epoch time of the last frame processed.
        """
        channel_1: List[int] = []
        channel_2: List[int] = []
        gain_list: List[int] = []
        last_epoch: Optional[int] = None

        for sensor_type, epoch_time, frame_counter, payload, metadata_bytes, sensor_status, checksum in self.frames():
            if sensor_type != 1:
                logger.debug("Skipping non-BCG frame %d of type %d", frame_counter, sensor_type)
                continue
            try:
                values = [v[0] for v in struct.iter_unpack("<i", payload[:-3])]
            except struct.error:
                logger.warning("frame %s: failed to unpack payload, skipping", frame_counter)
                continue

            if len(values) != 200:
                logger.warning("frame %s: unexpected sample count %d, skipping", frame_counter, len(values))
                continue

            channel_1.extend(values[:100])
            channel_2.extend(values[100:])
            gain_list.append(metadata_bytes)
            last_epoch = epoch_time

        return channel_1, channel_2, gain_list, last_epoch