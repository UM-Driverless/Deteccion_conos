import can
import time

import can_constants

class CAN():
    def __init__(self, logger=None):
        super().__init__()
        self.sleep_between_msg = 0.000
        self.logger = logger
        self.init_can()

    def init_can(self):
        """
        Initialize CAN.
        """
        self.bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=1000000)  # bustype = 'can'
        #self.logger.write_can_log(self.bus.channel_info)

        ## Implementamos la clase abstracta Listener (https://python-can.readthedocs.io/en/master/listeners.html)
        # De momento usamos BufferedReader, que es una implementación por defecto de la librería, con un buffer de mensajes. Si necesitamos que sea async podemos
        # usar AsyncBufferedReader.
        self.buffer = can.BufferedReader()
        self.notifier = can.Notifier(self.bus, [self.buffer])
        #self.logger.write_can_log("CAN listener connected")
        # TODO: [Sergio] Mover inicialización de la dirección a otra funcción
        self._init_steering()

    def _init_steering(self):
        data_steer_msg_n = [can_constants.STEER_INIT_ID['MSG_30'], can_constants.STEER_INIT_ID['MSG_31'],
                            can_constants.STEER_INIT_ID['MSG_32'], can_constants.STEER_INIT_ID['MSG_33'],
                            can_constants.STEER_INIT_ID['MSG_34'], can_constants.STEER_INIT_ID['MSG_35']
                            ]
        # self.logger.write_can_log("Init steering message -> " + str(data_steer_msg_n))
        self.send_message(can_constants.STEER_ID['STEER_ID'], 6, data_steer_msg_n)
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes

        data_steer_msg_a = [can_constants.STEER_INIT_ID['MSG_00'], can_constants.STEER_INIT_ID['MSG_01'],
                            can_constants.STEER_INIT_ID['MSG_02'], can_constants.STEER_INIT_ID['MSG_03'],
                            can_constants.STEER_INIT_ID['MSG_04']
                            ]
        # self.logger.write_can_log("Init steering message -> " + str(data_steer_msg_a))
        self.send_message(can_constants.STEER_ID['STEER_ID'], 5, data_steer_msg_a)
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes

        data_steer_msg_c = [can_constants.STEER_INIT_ID['MSG_20'], can_constants.STEER_INIT_ID['MSG_21'],
                            can_constants.STEER_INIT_ID['MSG_22'], can_constants.STEER_INIT_ID['MSG_23'],
                            can_constants.STEER_INIT_ID['MSG_24'], can_constants.STEER_INIT_ID['MSG_25']
                            ]
        # self.logger.write_can_log("Init steering message -> " + str(data_steer_msg_c))
        self.send_message(can_constants.STEER_ID['STEER_ID'], 6, data_steer_msg_c)
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes

    def send_message(self, idcan, datalength, data):
        #############################################################
        #    Sen CAN message
        #############################################################
        # DJU 17.02.2022         msg = can.Message(arbitration_id=can_constants.TRAJ_ID['TRAJ_ACT'], data=data, extended_id=False)
        # print(data)
        msg = can.Message(arbitration_id=idcan, data=data)
        #self.logger.write_can_log("MSG: " + str(msg))

        # self.logger.write_can_log("MSG: " + msg.__str__())

        try:
            self.bus.send(msg, timeout=can_constants.CAN_SEND_MSG_TIMEOUT)
        except can.CanError as e:
            print('Error al mandare msg CAN')
            error = e
            if hasattr(e, 'message'):
                error = e.message
            #self.logger.write_can_log("Sending ERROR: " + str(error))

if __name__ == "__main__":
    buscan = CAN()

