# TODO is this a module, a library, package or what? should we create an init file?

import numpy as np
from canlib import canlib, Frame
from globals.globals import * # Global variables and constants, as if they were here
from globals.can_constants import *
import time
import codecs

class Can_communication():
    """
    Class that manages the CAN communications with KVASER. Send and receive messages using KVaser for now.
    # https://pycanlib.readthedocs.io/en/v1.22.565/canlib/introduction.html
    
    # THREADING - https://www.kvaser.com/canlib-webhelp/page_user_guide_threads.html
    - Several threads can interact simultaneously with the same channel, but each thread must have a different handle.
    - That is, a different object.
    - self.ch is common between all the methods but different between the objects of the Can_communication class.
    - One object will be used for reading messages continuously in a thread, and another for sending messages in the main loop
    
    """
    
    # Print channels found
    num_channels = canlib.getNumberOfChannels()
    print(f'    Found {num_channels} CAN channels:')
    for channel in range(num_channels):
        ch_data = canlib.ChannelData(channel)
        print(f'        {channel}. {ch_data.channel_name} ({ch_data.card_upc_no} / {ch_data.card_serial_no})')
    
    
    def __init__(self, channel=0):
        """
        Initializes the Can_communication object.
        - Print message with the channels found
        - Open the channel
        - Set the acceptance filters
        - Configure
        - Activate chip
        
        Sets the self.ch variable, which is common for all the methods, but different between objects.
        """
        # print('Initializing CAN object from class Can_communication...')
        
        self.frame = -1
        
        # Open the CAN Channel
        self.ch = canlib.openChannel(channel=channel,flags=0,bitrate=canlib.Bitrate.BITRATE_1M)
        
        # Acceptance Filters - Hardware Level filters to reduce load - https://pycanlib.readthedocs.io/en/latest/canlib/sendandreceive.html#reading-messages
        # If the bits where the mask = 1 match with the code, the message is accepted.
        # ch.canAccept(0b1111,flag=canlib.AcceptFilterFlag.SET_MASK_STD)
        # ch.canAccept(0b0001,flag=canlib.AcceptFilterFlag.SET_CODE_STD)        
        
        # Set the CAN bus driver type (Silent, Normal...)
        self.ch.setBusOutputControl(canlib.Driver.NORMAL)
        
        # Activate the CAN chip
        self.ch.busOn()

        # LINK ID TO EACH PARSE METHOD FOR CAN MESSAGES. FOR NOW IN __INIT__ SO THE METHODS ARE ACCESSIBLE.
        self.CAN_ID_TO_METHODS = {
            int("300", 16): self.parse_senfl_imu,
            int("301", 16): self.parse_senfl_sig, # Various signals. Speed and wheel turns
            int("302", 16): self.parse_senfl_state,
            
            int("303", 16): self.parse_senfr_imu,
            int("304", 16): self.parse_senfr_sig,
            int("305", 16): self.parse_senfr_state,
            
            int("320", 16): self.parse_traj_act,
            int("321", 16): self.parse_traj_gps,
            int("322", 16): self.parse_traj_imu,
            int("323", 16): self.parse_traj_state,
            
            # Steer ... TODO in the future
            
            # ASSIS
            int("350", 16): self.parse_assis_c,
            int("351", 16): self.parse_assis_r,
            int("352", 16): self.parse_assis_l,
            
            int("360", 16): self.parse_asb_analog,
            int("361", 16): self.parse_asb_signals,
            int("362", 16): self.parse_asb_state,
            
            int("201", 16): self.parse_arduino,
        }

    def new_state(self, car_state):
        """
        Takes the car_state dictionary, its lastest received frame, and returns the updated version of car_state dictionary.
        
        The thread can send it through can_queue, to manage concurrency, and update car_state global variable in the main loop.
        """
        # PROCESS FRAME - https://pycanlib.readthedocs.io/en/latest/canlib/canframes.html
        id = self.frame.id
        data = [byte for byte in self.frame.data]
        dlc = self.frame.dlc
        timestamp = self.frame.timestamp
        # flags = self.frame.flags
        car_state = self.run_method(id, car_state)
        
        
        # TODO Get id etc, then look how to compare with ids from can_constants
        
        return car_state
    
    def receive_frame(self):
        # Receive the message
        self.frame = self.ch.read(timeout=1000) # TODO: IF NOT MESSAGES it returns an error, is that ok?

    def send_frame(self, data=0):
        # TODO IF CAN'T SEND THE MESSAGE, ERROR MESSAGE BUT NOT EXCEPTION? or ok ERROR
        # Transmit the message
        frame = Frame(id_=COMPUTER_CAN_ID, data = [0,0,0,0,0,0] , dlc=6) # data = [1,2] , data=b'HELLO!'
        self.ch.write(frame)

        # Wait until sent, timeout of 500ms
        self.ch.writeSync(timeout=500)
        
        # print(f'FRAME SENT: {frame}')
        
    def turn_off(self):
        # Chip OFF
        self.ch.busOff()
        
        # Close channel
        self.ch.close()
    
    def run_method(self, id, car_state):
        """
        Returns a method according to the id.
        Each CAN message requires different processing according to its id.
        
        id is a natural integer
        """
        # If the id is somewhere in the dictionary
        # TODO can just use get, and not check whether the id is inside? Use error value from get or something
        if id in self.CAN_ID_TO_METHODS.keys():
            # Run the method. self.CAN_ID_TO_METHODS.get(id) is a method
            car_state = self.CAN_ID_TO_METHODS.get(id)(car_state)
        else:
            print(f'ERROR: CANNOT RECOGNIZE CAN MESSAGE ID: 0x{np.base_repr(id,base=16)} (9_{id})')
            
        return car_state
    
    # METHODS PER CAN MESSAGE
    def parse_senfl_imu(self, car_state): # ID 0x300
        return car_state
    def parse_senfl_sig(self, car_state): # ID 0x301
        # 8 bytes - [Analog1 Analog2 Analog3 Digital Rel1 Rel2 Rel3]
        data = [byte for byte in self.frame.data]
        car_state['speed'] = data[4]*10 + data[5]
        return car_state
        
    def parse_senfl_state(self, car_state): # ID 0x302
        return car_state
        
    def parse_senfr_imu(self, car_state): # ID 0x303
        return car_state
    def parse_senfr_sig(self, car_state): # ID 0x304
        return car_state
    def parse_senfr_state(self, car_state): # ID 0x305
        return car_state
        
    def parse_traj_act(self, car_state): # ID 0x320
        return car_state
    def parse_traj_gps(self, car_state): # ID 0x321
        return car_state
    def parse_traj_imu(self, car_state): # ID 0x322
        return car_state
    def parse_traj_state(self, car_state): # ID 0x323
        return car_state
        
    # Steering TODO
        
    def parse_assis_c(self, car_state): # ID 0x350
        return car_state
    def parse_assis_r(self, car_state): # ID 0x351
        return car_state
    def parse_assis_l(self, car_state): # ID 0x352
        return car_state
        
    def parse_asb_analog(self, car_state): # ID 0x360
        return car_state
    def parse_asb_signals(self, car_state): # ID 0x361
        return car_state
    def parse_asb_state(self, car_state): # ID 0x362
        return car_state
        
    def parse_arduino(self, car_state): # ID 0x201
        return car_state