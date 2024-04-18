import os, sys, can, math, time, struct

from can_utils.can_constants import *

# TODO LOGGER

class CAN():
    '''
    __init__ initializes general can and steering module. Sets the startup values.
    '''
    def __init__(self, logger=None):
        super().__init__()
        self.sleep_between_msg = 0.003 # At least 1e-3s needed
        self.logger = logger
        self.init_can()
        self._init_steering()
        self.CAN_SEND_TIMEOUT = 0.005
        self.CAN_ACTION_DIMENSION = 100.
        self.MAXON_TOTAL_INCREMENTS = 122880. # 122880. Increments of maxon motor, from center, to get to the mechanical limit of the steering system at one side.

    def init_can(self):
        """
        Initialize CAN.
        """
        self.bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=1000000)
        self.buffer = can.BufferedReader()
        self.notifier = can.Notifier(self.bus, [self.buffer])
        ## Implementamos la clase abstracta Listener (https://python-can.readthedocs.io/en/master/listeners.html)
        # TODO READ THIS De momento usamos BufferedReader, que es una implementación por defecto de la librería, con un buffer de mensajes. Si necesitamos que sea async podemos
        # usar AsyncBufferedReader.
        
        # os.system("echo 0 | sudo -S ip link set down can0")
        # os.system("echo 0 | sudo -S ip link set can0 up")

    def send_message(self, id, data):
        '''
        Sends the CAN message with the id and the data. Like `cansend can0 {id}#{data}`
        
        Inputs:
            - (int) id: ID of maxon board
            - (array of ints) data: bytes to send
        '''
        msg = can.Message(arbitration_id=id, data=data, is_extended_id=False) # 11 bits of ID, instead of 29

        try:
            self.bus.send(msg, timeout=self.CAN_SEND_TIMEOUT)
        except can.CanError as e:
            print('Error al mandar el msg CAN: \n{e.message}')
        else:
            pass
            # print(f'Message sent ({self.bus.channel_info}): {hex(id)[2:]}#{data}')

    def send_action_msg(self,actuation): # TODO FIX, MANY MODIFICATIONS, VARIABLES REMOVED
        """
        Send the actions of actuation through CAN
        """
        
        self._steering_act(actuation)
        # self.send_message(601,int(actuation['throttle'] * CAN_ACTION_DIMENSION) # Para pasar rango de datos a 0:100
        # self.send_message(601,int(actuation['brake'] * CAN_ACTION_DIMENSION)
        
    def send_status_msg(self):
        """
        Send the status of the system through CAN message.

        Params to be defined. TODO CALCULATE STATUS. CHECK ASF AND DATALOGGER FOR REQUIREMENTS
        """
    def _init_steering(self):
        '''
        Set maxon board prepared to move
        
        1. DISABLE_POWER - cansend can0 601#2B40600600
        2. ENABLE_POWER - cansend can0 601#2B40600F00
        3. PROFILE_POSITION - cansend can0 601#2F60600001
        4. [OPTIONAL] SET_PARAMETERS - cansend can0 601#2360600001000000
        '''

        # 1. DISABLE_POWER - cansend can0 601#2B40600600
        self.send_message(CAN_IDS['STEER']['MAXON_ID'],CAN_MSG['STEER']['DISABLE_POWER'])
        time.sleep(self.sleep_between_msg)  # Steering controller needs 1ms between messages
        
        # 2. ENABLE_POWER - cansend can0 601#2B40600F00
        self.send_message(CAN_IDS['STEER']['MAXON_ID'],CAN_MSG['STEER']['ENABLE_POWER'])
        time.sleep(self.sleep_between_msg)  # Steering controller needs 1ms between messages

        # 3. PROFILE_POSITION - cansend can0 601#2F60600001
        self.send_message(CAN_IDS['STEER']['MAXON_ID'],CAN_MSG['STEER']['PROFILE_POSITION'])
        time.sleep(self.sleep_between_msg)  # Steering controller needs 1ms between messages
        
        # 4. [OPTIONAL] SET_PARAMETERS - cansend can0 601#2360600001000000
        # self.send_message(CAN_IDS['STEER']['MAXON_ID'],CAN_MSG['STEER']['SET_PARAMETERS'])
        # time.sleep(self.sleep_between_msg)  # Steering controller needs 1ms between messages
        
        

        # TODO MOVE CAN TO A THREAD.

    def _steering_act(self, actuation):
        '''
        Sends the steering messages needed to move after _init_steering() has been run
        
        1. SET_TARGET_POS - cansend can0 601#237A600000E00100
        2. MOVE_ABSOLUTE_POS - cansend can0 601#2B4060003F00
        3. TOGGLE_NEW_POS - cansend can0 601#284060000F00
        '''
        
        target_pos_bytes = self._s32_to_4_bytes(int(actuation['steer'] * self.MAXON_TOTAL_INCREMENTS))
        # print(f'target_pos: {target_pos_bytes}')
        print(f"agen_act(steer): {actuation['steer']}")
        print(f"Target value increments: {actuation['steer']*self.MAXON_TOTAL_INCREMENTS}\n")

        self.send_message(CAN_IDS['STEER']['MAXON_ID'], CAN_MSG['STEER']['SET_TARGET_POS'] + target_pos_bytes)
        time.sleep(self.sleep_between_msg)  # Controlador dirección necesita 0.001 seg entre mensajes
        self.send_message(CAN_IDS['STEER']['MAXON_ID'], CAN_MSG['STEER']['MOVE_ABSOLUTE_POS'])
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
    def interpret_message(self, msg_id):
        '''
        Do something with the message received
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
        
    # def decode_message(self, msg):
    #     '''
    #     Takes msg (string with hex)
    #     '''
    #     message = msg.data.hex() # Extrae los datos del mensaje CAN y los convierte a hexadecimal
    #     message = [int(message[index:index + 2], 16) for index in
    #                range(0, len(message), 2)] # Divide el mensaje can en 8 bytes y los convierte a decimal
    #     msg_id = int(hex(msg.arbitration_id), 16) # ID del mensaje CAN en string hexadecimal # TODO WTF
        
    #     return msg_id, message
    
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

    def _s32_to_4_bytes(self, signed_int):
        '''
        Takes a signed integer of 32 bits f, separates it (with float-32 representation) into bytes, and returns an array of ints with the bytes

        '''
        binary_representation = struct.pack('i', signed_int)
        # print(f'binary rep: {binary_representation}')

        int_array = [byte for byte in binary_representation]
        # print(f'int array: {int_array}')

        return int_array
