"""
Control the flow of information from the sensor (Camera and CAN data) to the main python program.
"""

from connection_utils.comunication_base import ComunicationInterface
from connection_utils.communication_controllers import zed, can_utils, zed_dummy
import numpy as np

class ConnectionManager(ComunicationInterface):
    def __init__(self, logger=None):
        self.logger = logger
        #self.camera = zed.Camera(logger=logger)  # Todavía no se ha implementado, de momento no hace nada.
        self.can = can_utils.CAN(logger=logger)  # Todavía no se ha implementado, de momento no hace nada.

    def get_data(self, params=None, verbose=0):
        """
        Return data coming from car or simulator sensors like camera, speed, gps, throttle, steer, brake, status msg
        :param params: a definir
        :param verbose: (int) If 0 no debug information is printed. If 1, print some debug info. If 2 print all debug info.
        :return: image, speed, throttle, steer, brake, clutch, gear ... (Ampliable a lo que necesitemos)
        """
        #image = self.camera.get_frame(self.camera.cam)

        can_msg = self.can.get_sensors_data()

        # Modificar valores, puestos a cero para que funcione el programa únicamente.
        speed = 0.
        throttle = 0.
        steer = 0.
        brake = 0.
        clutch = 0.
        gear = 0.
        rpm = 0.
        # [speed, throttle, steer, brake, gear ...]
        #return np.array(image), speed, throttle, steer, brake, clutch, gear, rpm
        return speed, throttle, steer, brake, clutch, gear, rpm

    def send_actions(self, throttle, steer, brake, clutch, upgear, downgear):
        """
        Send the actions to performs to the car actuators or to simulator.
        """
        print('Send throttle: ', throttle, ' clutch: ', clutch, ' brake: ', brake, ' steer: ', steer,  ' upgear: ', upgear, ' downgear: ', downgear)
        self.can.send_action_msg(throttle, brake, steer, clutch, upgear, downgear)

class ConnectionManager_dummy(ComunicationInterface):
    def __init__(self, logger=None):
        self.logger = logger
        self.camera = zed_dummy.Camera(logger=logger)
        self.can = can_utils.CAN_dummy(logger=logger)
        self.rpm = 100.

    def get_data(self, params=None, verbose=0):
        """
        Return data coming from car or simulator sensors like camera, speed, gps, throttle, steer, brake, status msg
        :param params: a definir
        :param verbose: (int) If 0 no debug information is printed. If 1, print some debug info. If 2 print all debug info.
        :return: image, speed, throttle, steer, brake, clutch, gear ... (Ampliable a lo que necesitemos)
        """
        image = self.camera.get_frame()
        can_msg = self.can.get_sensors_data()

        # Modificar valores, puestos a cero para que funcione el programa únicamente.
        self.rpm += 200
        speed = 0.
        throttle = 0.
        steer = 0.
        brake = 0.
        clutch = 0.
        gear = 0.
        # [speed, throttle, steer, brake, gear ...]
        return np.array(image), speed, throttle, steer, brake, clutch, gear, self.rpm

    def send_actions(self, throttle, steer, brake, clutch, upgear, downgear):
        """
        Send the actions to performs to the car actuators or to simulator.
        """
        self.can.send_action_msg(throttle, brake, steer, clutch, upgear, downgear)
