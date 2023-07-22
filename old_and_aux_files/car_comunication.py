"""
Control the flow of information from the sensor (Camera and CAN data) to the main python program.
"""
from connection_utils.communication_controllers import can_utils
import numpy as np

class ConnectionManager():
    """
    TODO clear description
    """
    
    def __init__(self, logger=None):
        self.logger = logger
        self.can = can_utils.CAN(logger=logger)  # Todavía no se ha implementado, de momento no hace nada.

    def get_data(self, params=None, verbose=0):
        """
        Return data coming from car or simulator sensors like camera, speed, gps, throttle, steer, brake, status msg
        :param params: a definir
        :param verbose: (int) If 0 no debug information is printed. If 1, print some debug info. If 2 print all debug info.
        :return: image, speed, throttle, steer, brake ... (Ampliable a lo que necesitemos)
        """
        can_msg = self.can.get_sensors_data()

        # Modificar valores, puestos a cero para que funcione el programa únicamente.
        speed = 0.
        throttle = 0.
        steer = 0.
        brake = 0.
        # [speed, throttle, steer, brake ...]
        #return np.array(image), speed, throttle, steer, brake
        return speed, throttle, steer, brake