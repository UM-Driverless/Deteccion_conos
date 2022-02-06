from abc import ABCMeta, abstractmethod


class CANInterface(object, metaclass=ABCMeta):
    @abstractmethod
    def init_can(self):
        """
        Initialize the CAN parameters.
        """

    @abstractmethod
    def get_sensors_data(self):
        """
        Returns the current data from sensors.
        :return: numpy nd array of floats
        """

    @abstractmethod
    def send_action_msg(self, throttle, brake, steer, clutch, gear):
        """
        Send the actions through CAN message.

        :param throttle: (float in [0., 1.]) Normalized throttle value.
        :param brake: (float in [0., 1.]) Normalized brake value.
        :param steer: (float in [-1., 1.]) Normalized steer. Range of steering in range [-180, 180]. ### Se puede cambiar si esto varía ###
        :param clutch: (float in [0., 1.]) Normalized clutch value.
        :param gear: (int) Gear to engage.
        :param upgear: (bool) True means up a gear, False do nothing.
        :param downgear: (bool) True means down a gear, False do nothing.
        """

    @abstractmethod
    def send_status_msg(self):
        """
        Send the status of the system through CAN message.

        Params to be defined.
        """

    @abstractmethod
    def send_sync_msg(self):
        """
        Send the synchronization values through CAN message.

        Params to be defined.
        """