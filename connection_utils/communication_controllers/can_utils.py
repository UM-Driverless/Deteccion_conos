import can
import math
import time
import os

from globals.globals import * # Global variables and constants, as if they were here

# TODO LOGGER

class CAN():
    '''
    __init__ initializes steering and can. Sets startup values
    
    
    '''
    def __init__(self, logger=None):
        super().__init__()
        self.sleep_between_msg = 0.002
        self.logger = logger
        self.init_can()
        self._init_steering()

    def init_can(self):
        """
        Initialize CAN.
        """
        self.bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=1000000)  # bustype = 'kvaser'
        # self.logger.write_can_log(self.bus.channel_info)
        # os.system("echo 0 | sudo -S ip link set down can0")
        # os.system("echo 0 | sudo -S ip link set can0 up")

        ## Implementamos la clase abstracta Listener (https://python-can.readthedocs.io/en/master/listeners.html)
        # De momento usamos BufferedReader, que es una implementación por defecto de la librería, con un buffer de mensajes. Si necesitamos que sea async podemos
        # usar AsyncBufferedReader.
        self.buffer = can.BufferedReader()
        self.notifier = can.Notifier(self.bus, [self.buffer])
        # self.logger.write_can_log("CAN listener connected")
        # TODO: [Sergio] Mover inicialización de la dirección a otra funcción
        # self._init_steering()

    def send_message(self, id, data):
        '''
        Sends the CAN message with the id and the data. Like `cansend can0 {id}#{data}`
        
        Inputs:
            - (int) id: ID of maxon board
            - (array of ints) data: bytes to send
        '''
        msg = can.Message(arbitration_id=id, data=data, is_extended_id=False) # 11 bits of ID, instead of 29

        try:
            self.bus.send(msg, timeout=globals.CAN_SEND_TIMEOUT)
        except can.CanError as e:
            print('Error al mandar el msg CAN: \n{e.message}')

    def send_action_msg(self, throttle, brake, steer): # TODO FIX, MANY MODIFICATIONS, VARIABLES REMOVED
        """
        Send the actions through CAN message.

        - throttle: (float in [0., 1.]) Normalized throttle value.
        - brake: (float in [0., 1.]) Normalized brake value.
        
        """
        # Para pasar rango de datos a 0:100
        throttle = int(throttle * CAN_ACTION_DIMENSION)
        brake = int(brake * CAN_ACTION_DIMENSION)
        # steer = int(((steer * CAN_ACTION_DIMENSION) + CAN_ACTION_DIMENSION)/2)
        steer = -int(steer * CAN_STEER_DIMENSION)
        # enviar datos actuadores
        data = [throttle, brake, 0, 0, 0]
        # self.logger.write_can_log("Send actions message -> " + str(data))
        self.send_message(id = CAN_IDS['TRAJ']['ACT'], data=data)
        time.sleep(self.sleep_between_msg)

        self._steering_act()

    def send_status_msg(self):
        """
        Send the status of the system through CAN message.

        Params to be defined.
        """
    def _init_steering(self):
        '''
        Set maxon board prepared to move
        
        1. DISABLE_POWER - cansend can0 601#2B40600600
        2. PROFILE_POSITION - cansend can0 601#2F60600001
        3. [OPTIONAL] SET_PARAMETERS - cansend can0 601#2360600001000000
        4. ENABLE_POWER - cansend can0 601#2B40600F00
        '''
        
        # 1. DISABLE_POWER - cansend can0 601#2B40600600
        self.send_message(CAN_IDS['STEER']['MAXON_ID'],CAN_MSG['DISABLE_POWER'])
        time.sleep(self.sleep_between_msg)  # Steering controller needs 1ms between messages
        
        # 2. PROFILE_POSITION - cansend can0 601#2F60600001
        self.send_message(CAN_IDS['STEER']['MAXON_ID'],CAN_MSG['PROFILE_POSITION'])
        time.sleep(self.sleep_between_msg)  # Steering controller needs 1ms between messages
        
        # 3. [OPTIONAL] SET_PARAMETERS - cansend can0 601#2360600001000000
        # self.send_message(CAN_IDS['STEER']['MAXON_ID'],CAN_MSG['SET_PARAMETERS'])
        # time.sleep(self.sleep_between_msg)  # Steering controller needs 1ms between messages
        
        # 4. ENABLE_POWER - cansend can0 601#2B40600F00
        self.send_message(CAN_IDS['STEER']['MAXON_ID'],CAN_MSG['ENABLE_POWER'])
        time.sleep(self.sleep_between_msg)  # Steering controller needs 1ms between messages
        

        # TODO MOVE CAN TO A THREAD.

    def _steering_act(self):
        '''
        Sends the steering messages needed to move after _init_steering() has been run
        
        1. SET_TARGET_POS - cansend can0 601#237A600000E00100
        2. MOVE_ABSOLUTE_POS - cansend can0 601#2B4060003F00
        3. TOGGLE_NEW_POS - cansend can0 601#284060000F00
        '''
        
        self.send_message(CAN_IDS['STEER']['MAXON_ID'], CAN_MSG['STEER']['SET_TARGET_POS'])
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes
        self.send_message(CAN_IDS['STEER']['MAXON_ID'], CAN_MSG['STEER']['MOVE_RELATIVE_POS'])
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes
        self.send_message(CAN_IDS['STEER']['MAXON_ID'], CAN_MSG['STEER']['TOGGLE_NEW_POS'])
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes
        
        
        ''' TODO REMOVE
        slSteer = long (
        SteerL  = ( slSteer & 0x000000FF )
        SteerCL = ( slSteer & 0x0000FF00 )
        SteerCH = ( slSteer & 0x00FF0000 )
        SteerH  = ( slSteer & 0xFF000000 )
        dataSteer = [0x22, 0x40, 0x60, 0, SteerL, SteerCL, SteerCH, SteerH]
        '''
    def decode_message(self, msg):
        message = msg.data.hex() # Extrae los datos del mensaje CAN y los convierte a hexadecimal
        message = [int(message[index:index + 2], 16) for index in
                   range(0, len(message), 2)] # Divide el mensaje can en 8 bytes y los convierte a decimal
        msg_id = int(hex(msg.arbitration_id), 16) # ID del mensaje CAN en string hexadecimal # TODO WTF
        
        return msg_id, message

    def route_msg(self, msg_id):
        '''
        
        '''
        
        if msg_id == CAN_IDS['SENFL']['IMU']:
            pass
        elif msg_id == CAN_IDS['SENFL']['SIG']:
            pass
        elif msg_id == CAN_IDS['SENFL']['STATE']:
            pass
        
        elif msg_id == CAN_IDS['SENFR']['IMU']:
            pass
        elif msg_id == CAN_IDS['SENFR']['SIG']:
            pass
        elif msg_id == CAN_IDS['SENFR']['STATE']:
            pass
        
        elif msg_id == CAN_IDS['TRAJ']['ACT']:
            pass
        elif msg_id == CAN_IDS['TRAJ']['GPS']:
            pass
        elif msg_id == CAN_IDS['TRAJ']['IMU']:
            pass
        elif msg_id == CAN_IDS['TRAJ']['STATE']:
            pass
        
        elif msg_id == CAN_IDS['ARD_ID']:
            print('ID from arduino read')
        
        elif msg_id == CAN_IDS['STEER']['MAXON_ID']:
            pass
        elif msg_id == CAN_IDS['STEER']['MSG_A']:
            pass
        elif msg_id == CAN_IDS['STEER']['MSG_B']:
            pass
        elif msg_id == CAN_IDS['STEER']['MSG_C']:
            pass
        elif msg_id == CAN_IDS['STEER']['MSG_N']:
            pass
        
        elif msg_id == CAN_IDS['STEER']['MSG_F']:
            pass
        elif msg_id == CAN_IDS['STEER']['MSG_D']:
            pass
        elif msg_id == CAN_IDS['STEER']['MSG_E']:
            pass
        
        elif msg_id == CAN_IDS['ASSIS']['COCKPIT']:
            pass
        elif msg_id == CAN_IDS['ASSIS']['RIGHT']:
            pass
        elif msg_id == CAN_IDS['ASSIS']['LEFT']:
            pass
        
        

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