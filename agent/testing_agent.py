from agent.agent_base import AgentInterface

import numpy as np


class AgentActuatorsTest(AgentInterface):
    """
    Agent to test the actuators.

    Genrates a sin  wave to send to the actuators
    """
    def __init__(self, logger=None):
        super().__init__(logger=logger)

        # Generamos una secuencia de tiempos
        self.time = np.arange(0, 6.28, 0.1)
        self.iterator = 0
        self.gear = 0
        self.max_gear = 4
        self.min_gear = 0
        self.throttle = 0.
        self.brake = 1.
        self.steer = 0.
        self.clutch = 1.

    def _get_sin_values(self, clip=False):
        """
        Genera una onda sinusoidal
        """
        time = self.time[self.iterator]
        sin_amplitude = np.sin(time)

        if clip:
            sin_amplitude = (sin_amplitude * 0.5) + 0.5

        self.iterator += 1

        # Si sobrepasamos la ultima posición de self.time volvemos al principio.
        if self.iterator >= self.time.shape[0]:
            self.iterator = 0

        return sin_amplitude

    def _get_inv_sin_values(self, clip=False):
        """
        Genera una onda sinusoidal
        """
        time = self.time[self.iterator]
        sin_amplitude = 1 - np.sin(time)

        if clip:
            sin_amplitude = (sin_amplitude * 0.5)

        self.iterator += 1

        # Si sobrepasamos la ultima posición de self.time volvemos al principio.
        if self.iterator >= self.time.shape[0]:
            self.iterator = 0

        return sin_amplitude

    def _get_cos_values(self, clip=False):
        """
        Genera una onda sinusoidal
        """
        time = self.time[self.iterator]
        cos_amplitude = np.cos(time)

        if clip:
            cos_amplitude = (cos_amplitude * 0.5) + 0.5

        self.iterator += 1

        # Si sobrepasamos la ultima posición de self.time volvemos al principio.
        if self.iterator >= self.time.shape[0]:
            self.iterator = 0

        return cos_amplitude

    def get_action(self, program):
        """
        Calcula las acciones que hay que realizar

        :param program: ([int]) List with the selected actuators to try.
        :return: throttle, brake, steer, clutch, upgear, downgear
        """
        upgear = False
        downgear = False

        if 1 in program:  # steer checking
            self.steer = self._get_sin_values()
        if 2 in program:  # throttle checking
            self.throttle = self._get_inv_sin_values(True)
        if 3 in program:  # brake checking
            self.brake = self._get_cos_values(True)
        if 4 in program:  # clutch checking
            self.clutch = self._get_sin_values(True)
        if 5 in program:  # upgear checking
            upgear = self.gear < self.max_gear
            self.gear = np.clip(self.gear + 1, self.min_gear, self.max_gear)
        if 6 in program:  # downgear checking
            downgear = self.gear > self.min_gear
            self.gear = np.clip(self.gear -1, self.min_gear, self.max_gear)

        return self.throttle, self.brake, self.steer, self.clutch, self.gear, upgear, downgear
