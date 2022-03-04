import os
import time
import can
from globals import can_constants
import math
import struct
import can.interfaces.socketcan.socketcan
import select

CAN_FRAME_HEADER_STRUCT = struct.Struct("=IBB2x")

def _add_flags_to_can_id(message):
    can_id = message.arbitration_id
    if message.is_extended_id:
        log.debug("sending an extended id type message")
        can_id |= CAN_EFF_FLAG
    if message.is_remote_frame:
        log.debug("requesting a remote frame")
        can_id |= CAN_RTR_FLAG
    if message.is_error_frame:
        log.debug("sending error frame")
        can_id |= CAN_ERR_FLAG
        print("sending error frame")

    return can_id

def build_can_frame(msg):
    """ CAN frame packing/unpacking (see 'struct can_frame' in <linux/can.h>)
    /**
     * struct can_frame - basic CAN frame structure
     * @can_id:  the CAN ID of the frame and CAN_*_FLAG flags, see above.
     * @can_dlc: the data length field of the CAN frame
     * @data:    the CAN frame payload.
     */
    struct can_frame {
        canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags */
        __u8    can_dlc; /* data length code: 0 .. 8 */
        __u8    data[8] __attribute__((aligned(8)));
    };

    /**
    * struct canfd_frame - CAN flexible data rate frame structure
    * @can_id: CAN ID of the frame and CAN_*_FLAG flags, see canid_t definition
    * @len:    frame payload length in byte (0 .. CANFD_MAX_DLEN)
    * @flags:  additional flags for CAN FD
    * @__res0: reserved / padding
    * @__res1: reserved / padding
    * @data:   CAN FD frame payload (up to CANFD_MAX_DLEN byte)
    */
    struct canfd_frame {
        canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags */
        __u8    len;     /* frame payload length in byte */
        __u8    flags;   /* additional flags for CAN FD */
        __u8    __res0;  /* reserved / padding */
        __u8    __res1;  /* reserved / padding */
        __u8    data[CANFD_MAX_DLEN] __attribute__((aligned(8)));
    };
    """
    can_id = _add_flags_to_can_id(msg)
    flags = 0
    if msg.bitrate_switch:
        flags |= CANFD_BRS
    if msg.error_state_indicator:
        flags |= CANFD_ESI
    max_len = 64 if msg.is_fd else 8
    data = bytes(msg.data).ljust(max_len, b'\x00')
    return CAN_FRAME_HEADER_STRUCT.pack(can_id, msg.dlc, flags) + data

if __name__ == '__main__':
    # Inicializar conexiones
    bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=1000000)
    bus.flush_tx_buffer()
    #try:

    #for i in range(1):
    start_time = time.time()
		
    # resize actions
    throttle = 0.99
    brake = 0.99
    steer = 0.99
    clutch = 0.99
        
    throttle = math.trunc(throttle * can_constants.CAN_ACTION_DIMENSION)
    brake = math.trunc(brake * can_constants.CAN_ACTION_DIMENSION)
    steer = math.trunc(((steer * can_constants.CAN_ACTION_DIMENSION) + can_constants.CAN_ACTION_DIMENSION)/2)
    clutch = math.trunc(clutch * can_constants.CAN_ACTION_DIMENSION)
    print('Send actions: ', throttle, clutch, brake, steer)
    data = [0, 0, 0, 0, 0, 0, 0, 0]

    msg = can.Message(arbitration_id=int("320", 16), data=data, extended_id=False)
    print(msg)
    timeout = 0.001
    try:
        started = time.time()
        # If no timeout is given, poll for availability
        if timeout is None:
            timeout = 0
        time_left = timeout
        data = build_can_frame(msg)

        while time_left >= 0:
            # Wait for write availability
            ready = select.select([], [bus.socket], [], time_left)[1]
            if not ready:
                # Timeout
                break
            sent = bus._send_once(data, msg.channel)
            if sent == len(data):
                break
            # Not all data were sent, try again with remaining data
            data = data[sent:]
            time_left = timeout - (time.time() - started)

        #bus.send(msg)
    except can.CanError as e:
        error = e
        if hasattr(e, 'message'):
            error = e.message
            print("Sending ERROR: " + str(error))
        print('exception')

    bus.flush_tx_buffer()
    
    print("FPS: ", 1.0 / (time.time() - start_time))
    time.sleep(1)
    bus.shutdown()
