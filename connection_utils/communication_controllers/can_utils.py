import can
import math
import struct
import time
import os

from globals.globals import * # Global variables and constants, as if they were here

# TODO LOGGER

class CAN():
    def __init__(self, logger=None):
        super().__init__()
        self.sleep_between_msg = 0.000
        self.logger = logger
        self.init_can()
        self._init_steering()

    def init_can(self):
        """
        Initialize CAN.
        """
        self.bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=1000000)  # bustype = 'kvaser'
        # self.logger.write_can_log(self.bus.channel_info)
        os.system("echo 0 | sudo -S ip link set down can0")
        os.system("echo 0 | sudo -S ip link set can0 up")

        ## Implementamos la clase abstracta Listener (https://python-can.readthedocs.io/en/master/listeners.html)
        # De momento usamos BufferedReader, que es una implementación por defecto de la librería, con un buffer de mensajes. Si necesitamos que sea async podemos
        # usar AsyncBufferedReader.
        self.buffer = can.BufferedReader()
        self.notifier = can.Notifier(self.bus, [self.buffer])
        # self.logger.write_can_log("CAN listener connected")
        # TODO: [Sergio] Mover inicialización de la dirección a otra funcción
        # self._init_steering()

    def send_status_msg(self):
        """
        Send the status of the system through CAN message.

        Params to be defined.
        """
    def send_sync_msg(self):
        ''''''
    def _init_steering(self):
        # self.logger.write_can_log("Init steering message -> " + str(CAN['STEER']['MSG_N']))
        self.send_message(CAN['STEER']['ID'], 6, CAN['STEER']['MSG_N'])
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes

        # self.logger.write_can_log("Init steering message -> " + str(CAN['STEER']['MSG_A']))
        self.send_message(CAN['STEER']['ID'], 5, CAN['STEER']['MSG_A'])
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes

        # self.logger.write_can_log("Init steering message -> " + str(data_steer_msg_c))
        self.send_message(CAN['STEER']['ID'], 6, CAN['STEER']['MSG_C'])
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes

    def get_sensors_data(self):
        """
        Returns the current data from sensors.
        :return: numpy nd array of floats
        """
        pass
        # all_msg_received = False
        # while not all_msg_received:
        #    msg = self.buffer.get_message()  # Revisar si se puede configurar el timeout
        #    if msg is not None:
        #        # msg.channel.fetchMessage()
        #        print(msg)
        #        msg_id, message = self.decode_message(msg)
        #        print(hex(msg_id), message)
        #        all_msg_received = True
        #        self.route_msg(msg_id)
        #    # TODO: eliminar cuando se reciban los mensajes y configurar salida del bucle
        #    all_msg_received = True
        # Se tiene que poner a true cuando se hayan recibido los mensajes necesarios para calcular las acciones. Velocidad, posición actuadores, rpm...

        return 0

    def send_action_msg(self, throttle, brake, steer): # TODO FIX, MANY MODIFICATIONS, VARIABLES REMOVED
        """
        Send the actions through CAN message.

        :param throttle: (float in [0., 1.]) Normalized throttle value.
        :param brake: (float in [0., 1.]) Normalized brake value.
        """
        # Para pasar rango de datos de -1:1 a 0:100
        throttle = math.trunc(throttle * CAN_ACTION_DIMENSION)
        brake = math.trunc(brake * CAN_ACTION_DIMENSION)
        # steer = math.trunc(((steer * CAN_ACTION_DIMENSION) + CAN_ACTION_DIMENSION)/2)
        steer = -math.trunc(int(steer * CAN_STEER_DIMENSION))
        # print('Send actions: ', throttle, brake, steer)
        # enviar datos actuadores
        data = [throttle, brake, 0, 0, 0]
        # self.logger.write_can_log("Send actions message -> " + str(data))
        self.send_message(id = CAN['TRAJ']['ACT'], datalength=8, data=data)
        time.sleep(self.sleep_between_msg)

        # enviar datos steering
        # habilita la direccion
        # self.logger.write_can_log("Send actions message -> " + str(CAN['STEER']['MSG_A']))
        self.send_message(CAN['STEER']['ID'], 6, CAN['STEER']['MSG_A'])
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes

        # envia los datos
        self.send_message(CAN['STEER']['ID'], 8, CAN['STEER']['MSG_B'])
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes
        '''
        slSteer = long (
        SteerL  = ( slSteer & 0x000000FF )
        SteerCL = ( slSteer & 0x0000FF00 )
        SteerCH = ( slSteer & 0x00FF0000 )
        SteerH  = ( slSteer & 0xFF000000 )
        dataSteer = [0x22, 0x40, 0x60, 0, SteerL, SteerCL, SteerCH, SteerH]
        '''

        # ejecuta el movimiento
        # self.logger.write_can_log("Send actions message -> " + str(data_steer_msg_e))
        self.send_message(CAN['STEER']['ID'], 6, CAN['STEER']['MSG_C'])
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes. Esta espera no debería hacer falta.

    def send_message(self, id, datalength, data):
        msg = can.Message(arbitration_id=id, data=data, is_extended_id=False) # 11 bits of ID, instead of 29

        try:
            self.bus.send(msg, timeout=globals.CAN_SEND_TIMEOUT)
        except can.CanError as e:
            print('Error al mandar el msg CAN')
            error = e
            if hasattr(e, 'message'):
                error = e.message
            # self.logger.write_can_log("Sending ERROR: " + str(error))

    def decode_message(self, msg):
        message = msg.data.hex()  # Extrae los datos del mensaje CAN y los convierte a hexadecimal
        message = [int(message[index:index + 2], 16) for index in
                   range(0, len(message), 2)]  # Divide el mensaje can en 8 bytes y los convierte a decimal
        msg_id = int(hex(msg.arbitration_id), 16)  # ID del mensaje CAN en string hexadecimal

        return msg_id, message

    def route_msg(self, msg_id):
        print('route msg: ', msg_id, CAN['ARD_ID'])
        if msg_id in SEN_ID.values():
            if msg_id == CAN['SENFL']['IMU']:
                pass
            if msg_id == CAN['SENFL']['SIG']:
                pass
            if msg_id == CAN['SENFL']['STATE']:
                pass
        if msg_id in TRAJ_ID.values():
            pass
        if msg_id in ASSIS_ID.values():
            pass
        if msg_id in ASB_ID.values():
            pass
        if msg_id in ARD_ID.values():
            print('Message from arduino read')
            if msg_id == ARD_ID['ID']:
                print('ID from arduino read')

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