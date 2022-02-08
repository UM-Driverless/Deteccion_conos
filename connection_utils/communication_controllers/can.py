from connection_utils.communication_controllers.can_interface import CANInterface
import can
from can import can_constants
import math


class CAN(CANInterface):
    def __init__(self, logger=None):
        super(CAN).__init__()
        self.logger = logger
        self.init_can()
        self.n_good_messages = 0  # TODO: variable para depurar la recepción de mensajes CAN. Eliminar cuando ya este testeado

    def init_can(self):
        """
        Initialize CAN.
        """
        self.bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=1000000)
        self.logger.write_can_log(self.bus.channel_info)

        ## Implementamos la clase abstracta Listener (https://python-can.readthedocs.io/en/master/listeners.html)
        # De momento usamos BufferedReader, que es una implementación por defecto de la librería, con un buffer de mensajes. Si necesitamos que sea async podemos
        # usar AsyncBufferedReader.
        self.buffer = can.BufferedReader()
        self.notifier = can.Notifier(self.bus, [self.buffer])
        self.logger.write_can_log("CAN listener connected")

    def get_sensors_data(self):
        """
        Returns the current data from sensors.
        :return: numpy nd array of floats
        """
        msg = self.buffer.get_message()

        if msg is not None:
            self.logger.write_can_log("Message well received (" + str(self.n_good_messages + 1) + ") -> " + str(msg))
            # self.logger.write_can_log("Mensaje bueno recibido (" + str(self.n_good_messages + 1) + ") -> " + msg.__str__())

            self.n_good_messages += 1

            message = self.decode_message(msg)
            self.logger.write_can_log("Message read (" + str(message) + ") ")

            # TODO: No se que checkea esta condición
            if msg.arbitration_id == can_constants.SIG_SENFL or msg.arbitration_id == can_constants.SIG_SENFR or msg.arbitration_id == can_constants.SIG_SENRL or msg.arbitration_id == can_constants.SIG_SENRR:
                # Ponía llamada a la red neuronal, pero eso no va a ir aquí
                pass
        return message

    def send_action_msg(self, throttle, brake, steer, clutch, upgear, downgear):
        # TODO: el cuarto valor son las marchas, es mejor mandar la señal  upgear, downgear o la marcha en si.
        """
        Send the actions through CAN message.

        :param throttle: (float in [0., 1.]) Normalized throttle value.
        :param brake: (float in [0., 1.]) Normalized brake value.
        :param steer: (float in [-1., 1.]) Normalized steer.
        :param clutch: (float in [0., 1.]) Normalized clutch value.
        :param upgear: (bool) True means up a gear, False do nothing.
        :param downgear: (bool) True means down a gear, False do nothing.
        """
        data = [math.trunc(throttle), math.trunc(brake), math.trunc(clutch), math.trunc(steer), 1, 0, 0, 0]
        self.logger.write_can_log("Send actions message -> " + str(data))
        self.send_message(data)

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

    def send_message(self, data):
        print('Send message: dummy function')

        #############################################################
        #    Sen CAN message
        #############################################################
        msg = can.Message(arbitration_id=can_constants.TRAJECTORY_ACT, data=data, extended_id=False)
        self.logger.write_can_log("MSG: " + str(msg))

        # self.logger.write_can_log("MSG: " + msg.__str__())

        try:
            self.bus.send(msg)
        except can.CanError as e:
            error = e
            if hasattr(e, 'message'):
                error = e.message
            self.logger.write_can_log("Sending ERROR: " + error)


    def decode_message(self, msg):
        # TODO: Esta funcion que hace? las onstantes ya están definidas en can_constants
        sender_id = msg.arbitration_id
        switcher = {
            # Rueda delantera izquierda
            can_constants.IMU_SENFL: self.get_sen_imu(0, msg.data),
            can_constants.SIG_SENFL: self.get_sen_signals(0, msg.data),
            can_constants.STATE_SENFL: "SEN STATE",
            # Rueda delantera derecha
            can_constants.IMU_SENFR: self.get_sen_imu(1, msg.data),
            can_constants.SIG_SENFR: self.get_sen_signals(1, msg.data),
            can_constants.STATE_SENFR: 0x305,
            # Rueda trasera izquierda
            can_constants.IMU_SENRL: self.get_sen_imu(2, msg.data),
            can_constants.SIG_SENRL: self.get_sen_signals(2, msg.data),
            can_constants.STATE_SENRL: 0x308,
            # Rueda trasera derecha
            can_constants.IMU_SENRR: self.get_sen_imu(3, msg.data),
            can_constants.SIG_SENRR: self.get_sen_signals(3, msg.data),
            can_constants.STATE_SENRR: 0x311,

            # ASSIS
            can_constants.ASSIS_C: 0x350,
            can_constants.ASSIS_R: 0x351,
            can_constants.ASSIS_L: 0x352,
            # ASB
            can_constants.ASB_ANALOG: 0x360,
            can_constants.ASB_SIGNALS: 0x361,
            can_constants.ASB_STATE: 0x362
            }
        return switcher.get(sender_id, lambda: "Código inválido")

    def get_sen_imu(self, wheel, data):
        acceleration_X = data[0]
        acceleration_Y = data[1]
        acceleration_Z = data[2]
        giroscope_X = data[3]
        giroscope_Y = data[4]
        giroscope_Z = data[5]
        magnet_X = data[6]
        magnet_Y = data[7]

        return [wheel, acceleration_X, acceleration_Y, acceleration_Z, giroscope_X, giroscope_Y, giroscope_Z, magnet_X,
                magnet_Y]

    def get_sen_signals(self, wheel, data):
        analog_1 = data[0]
        analog_2 = data[1]
        analog_3 = data[2]
        digital = data[3]

        speed_int = data[4]
        speed_decimals = data[5]
        speed = speed_int + speed_decimals / 10 ** len(str(speed_decimals))

        revolutions_1 = data[6]
        revolutions_2 = data[7]

        revolutions = revolutions_2 * 16 ** 2 + revolutions_1

        return [wheel, analog_1, analog_2, analog_3, digital, speed, revolutions]

class CAN_dummy(CANInterface):
    def __init__(self, logger=None):
        super(CAN).__init__()
        self.logger = logger
        self.init_can()
        self.n_good_messages = 0  # TODO: variable para depurar la recepción de mensajes CAN. Eliminar cuando ya este testeado

    def init_can(self):
        """
        Initialize CAN.
        """
        self.bus = None
        self.logger.write_can_log("Dummy CAN initialized")

        ## Implementamos la clase abstracta Listener (https://python-can.readthedocs.io/en/master/listeners.html)
        # De momento usamos BufferedReader, que es una implementación por defecto de la librería, con un buffer de mensajes. Si necesitamos que sea async podemos
        # usar AsyncBufferedReader.
        self.buffer = None
        self.notifier = None
        self.logger.write_can_log("Dummy CAN listener connected")

    def get_sensors_data(self):
        """
        Returns the current data from sensors.
        :return: numpy nd array of floats
        """
        msg = None

        self.logger.write_can_log("Dummy CAN Message well received")
        # self.logger.write_can_log("Mensaje bueno recibido (" + str(self.n_good_messages + 1) + ") -> " + msg.__str__())

        self.n_good_messages += 1

        message = None
        self.logger.write_can_log("Dummy CAN Message read")

        return message

    def send_action_msg(self, throttle, brake, steer, clutch, gear):
        pass

    def send_status_msg(self):
        pass

    def send_sync_msg(self):
        pass

    def send_message(self, data):
        pass

    def decode_message(self, msg):
        pass

    def get_sen_imu(self, wheel, data):
        pass

    def get_sen_signals(self, wheel, data):
        pass