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
    def __init__(self, max_data_to_store=25, print_can_data=False, print_agent_actions=True):
        """
        :param max_data_to_store: maximun number of time steps to save in memory
        """
        super(VisualizeActuators).__init__()

        self.throttle_historial = deque(maxlen=max_data_to_store)
        self.brake_historial = deque(maxlen=max_data_to_store)
        self.steer_historial = deque(maxlen=max_data_to_store)
        self.clutch_historial = deque(maxlen=max_data_to_store)
        self.gear_historial = deque(maxlen=max_data_to_store)
        self.rpm_historial = deque(maxlen=max_data_to_store)

        self.in_gear_historial = deque(maxlen=max_data_to_store)
        self.in_speed_historial = deque(maxlen=max_data_to_store)
        self.in_throttle_historial = deque(maxlen=max_data_to_store)
        self.in_brake_historial = deque(maxlen=max_data_to_store)
        self.in_steer_historial = deque(maxlen=max_data_to_store)
        self.in_clutch_historial = deque(maxlen=max_data_to_store)
        self.in_rpm_historial = deque(maxlen=max_data_to_store)

        self.print_can_data_flag = print_can_data
        self.print_agent_actions_flag = print_agent_actions
        self.init_plots(print_can_data=print_can_data, print_agent_actions=print_agent_actions)

    def visualize(self, can_data, agent_data, real_time=False):
        [in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm] = can_data
        [throttle, brake, steer, clutch, gear, rpm] = agent_data

        self.throttle_historial.append(throttle)
        self.brake_historial.append(brake)
        self.steer_historial.append(steer)
        self.clutch_historial.append(clutch)
        self.gear_historial.append(gear)
        self.rpm_historial.append(rpm)

        self.in_gear_historial.append(in_gear)
        self.in_speed_historial.append(in_speed)

        self.in_throttle_historial.append(in_throttle)
        self.in_brake_historial.append(in_brake)
        self.in_steer_historial.append(in_steer)
        self.in_clutch_historial.append(in_clutch)
        self.in_rpm_historial.append(in_rpm)

        # if real_time:
        #     plt.clf()
        if self.print_can_data_flag:
            self.print_can_data()
        if self.print_agent_actions_flag:
            self.print_agent_data()

        if real_time:
            plt.draw()
            plt.pause(0.01)
        else:
            plt.show()

    def init_plots(self, print_can_data=False, print_agent_actions=True):
        if print_agent_actions:
            self.agent_fig = plt.figure(1)

            self.agent_ax_1 = self.agent_fig.add_subplot(3, 3, 1)
            self.agent_ax_2 = self.agent_fig.add_subplot(3, 3, 2)
            self.agent_ax_3 = self.agent_fig.add_subplot(3, 3, 3)
            self.agent_ax_4 = self.agent_fig.add_subplot(3, 3, 4)
            self.agent_ax_5 = self.agent_fig.add_subplot(3, 3, 5)

        if print_can_data:
            self.can_fig = plt.figure(2)

            self.can_ax_1 = self.can_fig.add_subplot(3, 4, 1)
            self.can_ax_2 = self.can_fig.add_subplot(3, 4, 2)
            self.can_ax_3 = self.can_fig.add_subplot(3, 4, 3)
            self.can_ax_4 = self.can_fig.add_subplot(3, 4, 4)
            self.can_ax_5 = self.can_fig.add_subplot(3, 4, 5)
            self.can_ax_6 = self.can_fig.add_subplot(3, 4, 6)
            self.can_ax_7 = self.can_fig.add_subplot(3, 4, 7)

    def print_can_data(self):
        lin_space = np.linspace(0, len(self.throttle_historial), num=len(self.throttle_historial))

        self.can_ax_1.clear()
        self.can_ax_2.clear()
        self.can_ax_3.clear()
        self.can_ax_4.clear()
        self.can_ax_5.clear()
        self.can_ax_6.clear()
        self.can_ax_7.clear()

        self.can_ax_1.set(xlabel='steps', ylabel='force', title='CAN throttle')
        self.can_ax_2.set(xlabel='steps', ylabel='force', title='CAN brake')
        self.can_ax_3.set(xlabel='steps', ylabel='force', title='CAN steer')
        self.can_ax_4.set(xlabel='steps', ylabel='force', title='CAN clutch')
        self.can_ax_5.set(xlabel='steps', ylabel='force', title='CAN gear')
        self.can_ax_6.set(xlabel='steps', ylabel='force', title='CAN speed')
        self.can_ax_7.set(xlabel='steps', ylabel='rpm', title='CAN rpm')

        self.can_ax_1.plot(lin_space, self.in_throttle_historial)
        self.can_ax_2.plot(lin_space, self.in_brake_historial)
        self.can_ax_3.plot(lin_space, self.in_steer_historial)
        self.can_ax_4.plot(lin_space, self.in_clutch_historial)
        self.can_ax_5.plot(lin_space, self.in_gear_historial)
        self.can_ax_6.plot(lin_space, self.in_speed_historial)
        self.can_ax_7.plot(lin_space, self.in_rpm_historial)

        # lin_space = np.linspace(0, len(self.throttle_historial), num=len(self.throttle_historial))
        #
        # fig = plt.figure(2)
        #
        # ax = fig.add_subplot(3, 4, 1)
        # ax.plot(lin_space, self.in_throttle_historial)
        # ax.set(xlabel='steps', ylabel='force', title='CAN throttle')
        #
        # ax = fig.add_subplot(3, 4, 2)
        # ax.plot(lin_space, self.in_brake_historial)
        # ax.set(xlabel='steps', ylabel='force', title='CAN brake')
        #
        # ax = fig.add_subplot(3, 4, 3)
        # ax.plot(lin_space, self.in_steer_historial)
        # ax.set(xlabel='steps', ylabel='force', title='CAN steer')
        #
        # ax = fig.add_subplot(3, 4, 4)
        # ax.plot(lin_space, self.in_clutch_historial)
        # ax.set(xlabel='steps', ylabel='force', title='CAN clutch')
        #
        # ax = fig.add_subplot(3, 4, 5)
        # ax.plot(lin_space, self.in_gear_historial)
        # ax.set(xlabel='steps', ylabel='force', title='CAN gear')
        #
        # ax = fig.add_subplot(3, 4, 6)
        # ax.plot(lin_space, self.in_speed_historial)
        # ax.set(xlabel='steps', ylabel='force', title='CAN speed')
        #
        # ax = fig.add_subplot(3, 4, 7)
        # ax.plot(lin_space, self.in_rpm_historial)
        # ax.set(xlabel='steps', ylabel='rpm', title='CAN rpm')

    def print_agent_data(self):
        lin_space = np.linspace(0, len(self.throttle_historial), num=len(self.throttle_historial))
        self.agent_ax_1.clear()
        self.agent_ax_2.clear()
        self.agent_ax_3.clear()
        self.agent_ax_4.clear()
        self.agent_ax_5.clear()

        self.agent_ax_1.set(xlabel='steps', ylabel='force', title='throttle')
        self.agent_ax_2.set(xlabel='steps', ylabel='force', title='brake')
        self.agent_ax_3.set(xlabel='steps', ylabel='force', title='steer')
        self.agent_ax_4.set(xlabel='steps', ylabel='force', title='clutch')
        self.agent_ax_5.set(xlabel='steps', ylabel='force', title='gear')

        self.agent_ax_1.plot(lin_space, self.throttle_historial)
        self.agent_ax_2.plot(lin_space, self.brake_historial)
        self.agent_ax_3.plot(lin_space, self.steer_historial)
        self.agent_ax_4.plot(lin_space, self.clutch_historial)
        self.agent_ax_5.plot(lin_space, self.gear_historial)

        # fig = plt.figure(1)
        #
        # ax = fig.add_subplot(3, 3, 1)
        # ax.plot(lin_space, self.throttle_historial)
        # ax.set(xlabel='steps', ylabel='force', title='throttle')
        #
        # ax = fig.add_subplot(3, 3, 2)
        # ax.plot(lin_space, self.brake_historial)
        # ax.set(xlabel='steps', ylabel='force', title='brake')
        #
        # ax = fig.add_subplot(3, 3, 3)
        # ax.plot(lin_space, self.steer_historial)
        # ax.set(xlabel='steps', ylabel='force', title='steer')
        #
        # ax = fig.add_subplot(3, 3, 4)
        # ax.plot(lin_space, self.clutch_historial)
        # ax.set(xlabel='steps', ylabel='force', title='clutch')
        #
        # ax = fig.add_subplot(3, 3, 5)
        # ax.plot(lin_space, self.gear_historial)
        # ax.set(xlabel='steps', ylabel='force', title='gear')


    def close_windows(self):
        plt.close()

