import os
import time
import can
from globals import can_constants
import math

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
    try:
        bus.send(msg)
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
