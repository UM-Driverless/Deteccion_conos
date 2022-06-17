from connection_utils.communication_controllers.can_interface import CANInterface
import can
from globals import can_constants
import math
import struct
import time



class CAN(CANInterface):
    def __init__(self, logger=None):
        super().__init__()
        self.sleep_between_msg = 0.01
        self.time_traj = 0.0
        self.logger = logger
        self.init_can()

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
        # TODO: [Sergio] Mover inicialización de la dirección a otra funcción
        self._init_steering()
    
    def _init_steering(self):
        data_steer_msg_n = [can_constants.STEER_INIT_ID['MSG_30'], can_constants.STEER_INIT_ID['MSG_31'], can_constants.STEER_INIT_ID['MSG_32'], can_constants.STEER_INIT_ID['MSG_33'], can_constants.STEER_INIT_ID['MSG_34'], can_constants.STEER_INIT_ID['MSG_35']
]
        #self.logger.write_can_log("Init steering message -> " + str(data_steer_msg_n))
        self.send_message(can_constants.STEER_ID['STEER_ID'], 6, data_steer_msg_n)
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes

        data_steer_msg_a = [can_constants.STEER_INIT_ID['MSG_00'], can_constants.STEER_INIT_ID['MSG_01'], can_constants.STEER_INIT_ID['MSG_02'], can_constants.STEER_INIT_ID['MSG_03'], can_constants.STEER_INIT_ID['MSG_04']
] 
        #self.logger.write_can_log("Init steering message -> " + str(data_steer_msg_a))
        self.send_message(can_constants.STEER_ID['STEER_ID'], 5, data_steer_msg_a)
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes

        data_steer_msg_c = [can_constants.STEER_INIT_ID['MSG_20'], can_constants.STEER_INIT_ID['MSG_21'], can_constants.STEER_INIT_ID['MSG_22'], can_constants.STEER_INIT_ID['MSG_23'], can_constants.STEER_INIT_ID['MSG_24'], can_constants.STEER_INIT_ID['MSG_25']
] 
        #self.logger.write_can_log("Init steering message -> " + str(data_steer_msg_c))
        self.send_message(can_constants.STEER_ID['STEER_ID'], 6, data_steer_msg_c)
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes

    def get_amr(self):
        amr = 0
        all_msg_received = False
        while not all_msg_received:
            msg = self.buffer.get_message(0.1)  # Revisar si se puede configurar el timeout
            if msg is not None:
                # msg.channel.fetchMessage()
                msg_id, message = self.decode_message(msg)
                if msg_id == can_constants.STEERING_ID["STEERW_DV"]:
                    amr = message[0]
                all_msg_received = True
                self.route_msg(msg_id)
            # TODO: eliminar cuando se reciban los mensajes y configurar salida del bucle
            all_msg_received = True # Se tiene que poner a true cuando se hayan recibido los mensajes necesarios para calcular las acciones. Velocidad, posición actuadores, rpm...

        return amr

    def get_rpm(self):
        rpm_can = 0
        all_msg_received = False
        while not all_msg_received:
            msg = self.buffer.get_message(0.1)  # Revisar si se puede configurar el timeout
            if msg is not None:
                # msg.channel.fetchMessage()
                msg_id, message = self.decode_message(msg)
                if msg_id == can_constants.PMC_ID['PMC_ECU1']:
                    rpm_can = (message[1] << 8) | message[0]
                all_msg_received = True
                self.route_msg(msg_id)
            # TODO: eliminar cuando se reciban los mensajes y configurar salida del bucle
            all_msg_received = True # Se tiene que poner a true cuando se hayan recibido los mensajes necesarios para calcular las acciones. Velocidad, posición actuadores, rpm...

        return rpm_can

    def get_sensors_data(self):
        """
        Returns the current data from sensors.
        :return: numpy nd array of floats
        """
        all_msg_received = False
        while not all_msg_received:
            msg = self.buffer.get_message(0.1)  # Revisar si se puede configurar el timeout
            if msg is not None:
                #msg.channel.fetchMessage()
                print(msg)
                msg_id, message = self.decode_message(msg)
                print(hex(msg_id), message)
                if msg_id == can_constants.PMC_ID ['PMC_ECU1']:
                	rpm_can = (message[1] << 8) | message[0]
                if msg_id == can_constants.PMC_ID ['PMC_STATE']:
                	ASState = message[0]
                if msg_id == can_constants.SEN_ID['SIG_SENFL']:
                	speed_FL_can = message[4]
                if msg_id == can_constants.SEN_ID['SIG_SENFR']:
                	speed_FR_can = message[4]
                if msg_id == can_constants.SEN_ID['STEERW_DV']:
                	amr = message[0]
                if msg_id == can_constants.SEN_ID['ETC_STATE']:
                	clutch_state = message[2]
                all_msg_received = True
                self.route_msg(msg_id)
            # TODO: eliminar cuando se reciban los mensajes y configurar salida del bucle
            all_msg_received = True
                # Se tiene que poner a true cuando se hayan recibido los mensajes necesarios para calcular las acciones. Velocidad, posición actuadores, rpm...		  
        return 0

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
		#Para pasar rango de datos de -1:1 a 0:100
        throttle = math.trunc(throttle * can_constants.CAN_ACTION_DIMENSION)
        brake = math.trunc(brake * can_constants.CAN_ACTION_DIMENSION)
        #steer = math.trunc(((steer * can_constants.CAN_ACTION_DIMENSION) + can_constants.CAN_ACTION_DIMENSION)/2)
        steer = -math.trunc(int(steer * can_constants.CAN_STEER_DIMENSION))
        clutch = math.trunc(clutch * can_constants.CAN_CLUTCH_DIMENSION)
        upgear = math.trunc(upgear * can_constants.CAN_ACTION_DIMENSION)
        downgear = math.trunc(downgear * can_constants.CAN_ACTION_DIMENSION)
        print('Send actions: ', throttle, clutch, brake, steer, upgear, downgear)
        
        if ((time.time() - self.time_traj) > 0.00001):
        		print((time.time() - self.time_traj))
				#enviar datos actuadores
        		data = [throttle, clutch, brake, 0, upgear, downgear, 0, 0]
        		#self.logger.write_can_log("Send actions message -> " + str(data))
        		self.send_message(can_constants.TRAJ_ID['TRAJ_ACT'], 8, data)
        		time.sleep(self.sleep_between_msg)
        		self.time_traj = time.time()
        		
        		
        

		#enviar datos steering
        #habilita la direccion
        data_steer_msg_f = [can_constants.STEER_ID['MSG_00'], can_constants.STEER_ID['MSG_01'], can_constants.STEER_ID['MSG_02'], can_constants.STEER_ID['MSG_03'], can_constants.STEER_ID['MSG_04'], can_constants.STEER_ID['MSG_05']] 
        #self.logger.write_can_log("Send actions message -> " + str(data_steer_msg_f))
        self.send_message(can_constants.STEER_ID['STEER_ID'], 6, data_steer_msg_f)
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes

        #envia los datos
        ba = bytearray(struct.pack("i", steer))
        # steer_values = ["0x%02x" % b for b in ba]
        steer_values = [b for b in ba]
        #print('steer_values: ', steer_values)
        data_steer_msg_d = [can_constants.STEER_ID['MSG_10'], can_constants.STEER_ID['MSG_11'], can_constants.STEER_ID['MSG_12'], can_constants.STEER_ID['MSG_13'], steer_values[0], steer_values[1], steer_values[2], steer_values[3]]
        #self.logger.write_can_log("Send actions message -> " + str(data_steer_msg_d))
        self.send_message(can_constants.STEER_ID['STEER_ID'], 8, data_steer_msg_d)
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes
        '''
        slSteer = long (
        SteerL  = ( slSteer & 0x000000FF )
        SteerCL = ( slSteer & 0x0000FF00 )
        SteerCH = ( slSteer & 0x00FF0000 )
        SteerH  = ( slSteer & 0xFF000000 )
        dataSteer = [0x22, 0x40, 0x60, 0, SteerL, SteerCL, SteerCH, SteerH]
        '''

        #ejecuta el movimiento
        data_steer_msg_e = [can_constants.STEER_ID['MSG_20'], can_constants.STEER_ID['MSG_21'], can_constants.STEER_ID['MSG_22'], can_constants.STEER_ID['MSG_23'], can_constants.STEER_ID['MSG_24'], can_constants.STEER_ID['MSG_25']] #valores absolutos
        #self.logger.write_can_log("Send actions message -> " + str(data_steer_msg_e))
        self.send_message(can_constants.STEER_ID['STEER_ID'], 6, data_steer_msg_e)
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes. Esta espera no debería hacer falta.

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

    def send_message(self, idcan, datalength, data):
        #############################################################
        #    Sen CAN message
        #############################################################
		#DJU 17.02.2022         msg = can.Message(arbitration_id=can_constants.TRAJ_ID['TRAJ_ACT'], data=data, extended_id=False)
        print(data)
        msg = can.Message(arbitration_id=idcan, data=data, extended_id=False)
        self.logger.write_can_log("MSG: " + str(msg))

        # self.logger.write_can_log("MSG: " + msg.__str__())

        try:
            self.bus.send(msg, timeout=can_constants.CAN_SEND_MSG_TIMEOUT)
        except can.CanError as e:
            print('Error al mandare msg CAN')
            error = e
            if hasattr(e, 'message'):
                error = e.message
            self.logger.write_can_log("Sending ERROR: " + str(error))


    def decode_message(self, msg):
        message = msg.data.hex()  # Extrae los datos del mensaje CAN y los convierte a hexadecimal
        message = [int(message[index:index+2], 16) for index in range(0, len(message), 2)] # Divide el mensaje can en 8 bytes y los convierte a decimal
        msg_id = int(hex(msg.arbitration_id), 16) # ID del mensaje CAN en string hexadecimal

        return msg_id, message
        

    def route_msg(self, msg_id):
        print('route msg: ', msg_id, can_constants.ARD_ID)
        if msg_id in can_constants.SEN_ID.values():
             if msg_id == can_constants.SEN_ID['IMU_SENFL']:
                 pass
             if msg_id == can_constants.SEN_ID['SIG_SENFL']:
                 pass
             if msg_id == can_constants.SEN_ID['STATE_SENFL']:
                 pass
        if msg_id in can_constants.TRAJ_ID.values():
             pass
        if msg_id in can_constants.ASSIS_ID.values():
             pass
        if msg_id in can_constants.ASB_ID.values():
             pass
        if msg_id in can_constants.ARD_ID.values():
             print('Message from arduino readed')
             if msg_id == can_constants.ARD_ID['ID']:
                 print('ID from arduino readed')


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
        super().__init__()
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

    def send_action_msg(self, throttle, brake, steer, clutch, upgear, downgear):
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
