import struct
import logging
from typing import Iterator, Tuple, List, Optional
import numpy as np
import plotly.graph_objects as go


FRAME_SIZE  = 813
FSR_FRAME_SIZE = 15
FRAME_SIZE_TEM = 17
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
        
        with open(self.filename, "rb") as f:
            self.data = f.read()        

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
    
    def parse_header(self, blob: bytes, max_lines=3):
        header_lines = blob.split(b'\n', max_lines)[:max_lines]

        result = {}
        for line in header_lines:
            text = line.decode("utf-8", errors="strict").strip()
            if " : " in text:
                key, value = text.split(" : ", 1)
                result[key.strip()] = value.strip()

        return result
    
    def check_packet_loss(self, packets):
        """
        Checks packet loss based on an 8-bit sequence counter (0–255).

        :param packets: iterable of packets, each having a .seq attribute
        :return: list of tuples (index, lost_count)
        """
        prev_seq = None
        losses = []

        for index, packet in enumerate(packets):
            curr_seq = packet.seq

            if prev_seq is not None:
                expected = (prev_seq + 1) % 256
                if curr_seq != expected:
                    lost = (curr_seq - expected) % 256
                    losses.append((index, lost))

            prev_seq = curr_seq

        return losses



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
            # if len(frame) != FRAME_SIZE:
            #     logger.debug("Skipping frame with unexpected length %d", len(frame))
            #     continue
            if (len(frame) != FRAME_SIZE):
                if len(frame) != FSR_FRAME_SIZE:
                    if len(frame) != FRAME_SIZE_TEM:
                        continue

            # verify header (first 2 bytes)
            if struct.unpack_from("<H", frame, 0)[0] != struct.unpack_from("<H", MAGIC_HEADER, 0)[0]:
                logger.debug("Invalid header signature, skipping")
                continue

            sensor_type = frame[2]
            epoch_time = struct.unpack_from("<I", frame, 3)[0]
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
        fsr_values: List[int] = []
        fsr_time: List[int] = []
        cont_seq_counter: List[int] = []
        last_epoch: Optional[int] = None
        position = 0

        header_meta_data = self.parse_header(self.data)

        prev_seq = None
        previous_median_ch1 = None 
        previous_median_ch2 = None

        calculate_packet_loss = 0
        total_frames = 0
        epoch_time_list = []
        # prev = None
        for sensor_type, epoch_time, frame_counter, payload, metadata_bytes, sensor_status, checksum in self.frames():
            
            # if prev == None:
            #     prev = frame_counter
            # else:
            #     delta = frame_counter - prev
            #     if (delta != 1) | (delta != -255):
            lost = 0

            if sensor_type == 1:
                curr_seq = frame_counter
                if prev_seq is not None:
                    expected = (prev_seq + 1) % 256
                    if curr_seq != expected:
                        lost = (curr_seq - expected) % 256
                        logger.debug("Packet loss detected: %d packets lost", lost)

                prev_seq = curr_seq
                if lost > 0:
                    if previous_median_ch1 is not None:
                        for i in range(lost):
                            channel_1.extend([previous_median_ch1]*100)
                            channel_2.extend([previous_median_ch2]*100)
                try:
                    values = [v[0] for v in struct.iter_unpack("<i", payload[:-3])]
                    previous_median_ch1 = np.median(values[:100])
                    previous_median_ch2 = np.median(values[100:])
                except struct.error:
                    logger.warning("frame %s: failed to unpack payload, skipping", frame_counter)
                    continue

                if len(values) != 200:
                    logger.warning("frame %s: unexpected sample count %d, skipping", frame_counter, len(values))
                    continue

                channel_1.extend(values[:100])
                channel_2.extend(values[100:])
                calculate_packet_loss += lost
                gain_list.append(metadata_bytes)
            
            if sensor_type == 2:
                # print("----------")
                
                try:
                    fsr = struct.unpack("<H", payload[:-3])[0]
                    fsr_values.append(fsr)
                    fsr_time.append(epoch_time)
                    prev_seq = frame_counter
                except struct.error:
                    logger.warning("frame %s: failed to unpack payload for fsr, skipping", frame_counter)

            if sensor_type == 3:
                try:
                    # env_data = struct.unpack("<H", payload[:-3])[0]
                    prev_seq = frame_counter
                except struct.error:
                    logger.warning("frame %s: failed to unpack payload for env_data, skipping", frame_counter)
            

            
            epoch_time_list.append(epoch_time) 
            total_frames += 1



        return channel_1, channel_2, gain_list, epoch_time_list, header_meta_data, fsr_values, fsr_time, calculate_packet_loss, total_frames