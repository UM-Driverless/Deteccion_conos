from controller_agent.agent_base import AgentInterface
from trayectory_estimation.cone_processing import ConeProcessing

import numpy as np


class AgentActuatorsTest(AgentInterface):
    """
    Agent to test the actuators.

    Genrates a sin  wave to send to the actuators
    """
    def __init__(self):
        super(AgentActuatorsTest).__init__()

        # Generamos una secuencia de tiempos
        self.time = np.arange(0, 6.28, 0.1)
        self.iterator = 0

    def _get_sin_values(self, clip=False):
        """
        Genera una onda sinusoidal
        """
        time = self.time[self.iterator]
        sin_amplitude = np.sin(time)

        if clip:
            sin_amplitude = np.clip(sin_amplitude, a_min=0., a_max=1.)

        self.iterator += 1

        # Si sobrepasamos la ultima posición de self.time volvemos al principio.
        if self.iterator >= self.time.shape[0]:
            self.iterator = 0

        return sin_amplitude

    def get_action(self, program):
        """
        Calcula las acciones que hay que realizar

        :param program: (int) Select the testing program to execute.
        :return: throttle, brake, steer, clutch, upgear, downgear
        """
        throttle = 0.
        brake = 0.
        steer = 0.
        clutch = 0
        upgear = False
        downgear = False

        if program == 1:  # steer checking
            steer = self._get_sin_values()
        elif program == 2:  # throttle checking
            throttle = self._get_sin_values(True)
        elif program == 3:  # brake checking
            brake = self._get_sin_values(True)
        elif program == 4:  # clutch checking
            clutch = self._get_sin_values(True)
        elif program == 5:  # upgear checking
            upgear = True
        elif program == 6:  # downgear checking
            downgear = True

        return throttle, brake, steer, clutch, upgear, downgear