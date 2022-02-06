from abc import ABCMeta, abstractmethod

class ComunicationInterface(object, metaclass=ABCMeta):
    def __init__(self):
        self.imageWidth = None
        self.imageWidth = None

    @abstractmethod
    def get_data(self, params=None, verbose=0):
        """
        Return data coming from car or simulator sensors like camera, speed, gps, throttle, steer, brake, status msg
        :param params:
        :param verbose: (int) If 0 no debug information is printed. If 1, print some debug info. If 2 print all debug info.
        :return: image, speed, throttle, steer, brake, clutch, gear (Ampliable a lo que necesitemos)
        """

    @abstractmethod
    def send_actions(self, throttle, steer, brake, clutch, gear):
        """
        Send the actions to performs to the car actuators or to simulator.
        """
