from connection_utils.communication_controllers.can_interface import CANInterface


class CAN(CANInterface):
    def __init__(self):
        super(CAN).__init__()
        self.init_can()

    def init_can(self):
        """
        Initialize the CAN parameters.
        """
        pass

    def get_sensors_data(self):
        """
        Returns the current data from sensors.
        :return: numpy nd array of floats
        """
        # no implementado todavia
        return None

    def send_action_msg(self, throttle, brake, steer, clutch, upgear, downgear):
        """
        Send the actions through CAN message.

        :param throttle: (float in [0., 1.]) Normalized throttle value.
        :param brake: (float in [0., 1.]) Normalized brake value.
        :param steer: (float in [-1., 1.]) Normalized steer. Range of steering in range [-180, 180]. ### Se puede cambiar si esto varía ###
        :param clutch: (float in [0., 1.]) Normalized clutch value.
        :param upgear: (bool) True means up a gear, False do nothing.
        :param downgear: (bool) True means down a gear, False do nothing.
        """
        pass

    def send_status_msg(self):
        """
        Send the status of the system through CAN message.

        Params to be defined.
        """
        pass

    def send_sync_msg(self):
        """
        Send the synchronization values through CAN message.

        Params to be defined.
        """
        pass