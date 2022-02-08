import numpy as np

import warnings
warnings.filterwarnings("ignore")
from visualization_utils.visualize_base import VisualizeInterface
import matplotlib.pyplot as plt
from collections import deque

class VisualizeActuators(VisualizeInterface):
    """
    Intarfaz for visualization of the actuators data.
    """
    def __init__(self, max_data_to_store=10000):
        """
        :param max_data_to_store: maximun number of time steps to save in memory
        """
        super(VisualizeActuators).__init__()

        self.throttle_historial = deque(maxlen=max_data_to_store)
        self.brake_historial = []
        self.steer_historial = []
        self.clutch_historial = []
        self.gear_historial = []

        self.in_gear_historial = []
        self.in_speed_historial = []
        self.in_throttle_historial = []
        self.in_brake_historial = []
        self.in_steer_historial = []
        self.in_clutch_historial = []

    def visualize(self, can_data, agent_data, print_can_data=True, print_agent_actions=True, real_time=False):
        [in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear] = can_data
        [throttle, brake, steer, clutch, gear] = agent_data

        self.throttle_historial.append(throttle)
        self.brake_historial.append(brake)
        self.steer_historial.append(steer)
        self.clutch_historial.append(clutch)
        self.gear_historial.append(gear)

        self.in_gear_historial.append(in_gear)
        self.in_speed_historial.append(in_speed)

        self.in_throttle_historial.append(in_throttle)
        self.in_brake_historial.append(in_brake)
        self.in_steer_historial.append(in_steer)
        self.in_clutch_historial.append(in_clutch)

        if real_time:
            plt.clf()
        if print_can_data:
            self.print_can_data()
        if print_agent_actions:
            self.print_agent_data()

        if real_time:
            plt.draw()
            plt.pause(0.01)
        else:
            plt.show()

    def print_can_data(self):
        lin_space = np.linspace(0, len(self.throttle_historial), num=len(self.throttle_historial))

        fig = plt.figure(2)

        ax = fig.add_subplot(3, 3, 1)
        ax.plot(lin_space, self.in_throttle_historial)
        ax.set(xlabel='steps', ylabel='force', title='CAN throttle')

        ax = fig.add_subplot(3, 3, 2)
        ax.plot(lin_space, self.in_brake_historial)
        ax.set(xlabel='steps', ylabel='force', title='CAN brake')

        ax = fig.add_subplot(3, 3, 3)
        ax.plot(lin_space, self.in_steer_historial)
        ax.set(xlabel='steps', ylabel='force', title='CAN steer')

        ax = fig.add_subplot(3, 3, 4)
        ax.plot(lin_space, self.in_clutch_historial)
        ax.set(xlabel='steps', ylabel='force', title='CAN clutch')

        ax = fig.add_subplot(3, 3, 5)
        ax.plot(lin_space, self.in_gear_historial)
        ax.set(xlabel='steps', ylabel='force', title='CAN gear')

        ax = fig.add_subplot(3, 3, 6)
        ax.plot(lin_space, self.in_speed_historial)
        ax.set(xlabel='steps', ylabel='force', title='CAN speed')

    def print_agent_data(self):
        lin_space = np.linspace(0, len(self.throttle_historial), num=len(self.throttle_historial))

        fig = plt.figure(1)

        ax = fig.add_subplot(3, 3, 1)
        ax.plot(lin_space, self.throttle_historial)
        ax.set(xlabel='steps', ylabel='force', title='throttle')

        ax = fig.add_subplot(3, 3, 2)
        ax.plot(lin_space, self.brake_historial)
        ax.set(xlabel='steps', ylabel='force', title='brake')

        ax = fig.add_subplot(3, 3, 3)
        ax.plot(lin_space, self.steer_historial)
        ax.set(xlabel='steps', ylabel='force', title='steer')

        ax = fig.add_subplot(3, 3, 4)
        ax.plot(lin_space, self.clutch_historial)
        ax.set(xlabel='steps', ylabel='force', title='clutch')

        ax = fig.add_subplot(3, 3, 5)
        ax.plot(lin_space, self.gear_historial)
        ax.set(xlabel='steps', ylabel='force', title='gear')


    def close_windows(self):
        plt.close()

