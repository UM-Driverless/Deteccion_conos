import struct
import sys

from can_utils.can_utils import CAN

def float_to_bytes(f):
    '''
    Takes a float f, separates it into bytes, and returns an array of ints with the bytes

    '''
    # Use the 'f' format specifier to pack the float as a 32-bit float
    # Use '<' for little-endian byte order (change to '>' for big-endian)
    binary_representation = struct.pack('<f', f)

    int_array = [byte for byte in binary_representation]

    return int_array

pos = 0


can_bus = CAN()

while True:

    target_pos_bytes = float_to_bytes(pos)
    print(f'target_pos_bytes: {target_pos_bytes}')
    can_bus.send_message(0x601, target_pos_bytes)
    pos += 1e3