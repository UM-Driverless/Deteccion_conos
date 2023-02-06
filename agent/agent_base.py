from abc import ABCMeta, abstractmethod


class AgentInterface(object, metaclass=ABCMeta):
    def __init__(self, logger=None):
        self.logger = logger

    @abstractmethod
    def get_action(self, ref_point, img_center):
        """
        Calcular los valores de control
        :param ref_point: punto de referencia
        :param img_center: centro de la imagen
        :return: throttle, brake, steer, clutch, upgear, downgear
        """
