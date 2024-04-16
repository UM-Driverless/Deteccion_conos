# TODO is this a module, a library, package or what? should we create an init file?

# import numpy as np
from canlib import canlib, Frame # CanKvaser

class CanKvaser():
    """
    Class that manages the CAN communications with KVASER. Send and receive messages using KVaser for now.
    # https://pycanlib.readthedocs.io/en/v1.22.565/canlib/introduction.html
    
    # THREADING - https://www.kvaser.com/canlib-webhelp/page_user_guide_threads.html
    - Several threads can interact simultaneously with the same channel, but each thread must have a different handle.
    - That is, a different object.
    - self.ch is common between all the methods but different between the objects of the CanKvaser class.
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
        Initializes the CanKvaser object.
        - Print message with the channels found
        - Open the channel
        - Set the acceptance filters
        - Configure
        - Activate chip
        
        Sets the self.ch variable, which is common for all the methods, but different between objects.
        """
        # print('Initializing CAN object from class CanKvaser...')
        
        self.frame = -1
        
        # Open the CAN Channel
        self.ch = canlib.openChannel(channel=channel,flags=0,bitrate=canlib.Bitrate.BITRATE_1M)
        
        # Acceptance Filters - Hardware Level filters to reduce load - https://pycanlib.readthedocs.io/en/latest/canlib/sendandreceive.html#reading-messages
        # If the bits where the mask = 1 match with the code, the message is accepted.
        self.ch.canAccept(0b01100000000,flag=canlib.AcceptFilterFlag.SET_MASK_STD)
        self.ch.canAccept(0b01100000001,flag=canlib.AcceptFilterFlag.SET_CODE_STD)
        
        # TODO solve 0x7b message appearing when it shouldn't
        
        # Set the CAN bus driver type (Silent, Normal...)
        self.ch.setBusOutputControl(canlib.Driver.NORMAL)
        
        # Activate the CAN chip
        self.ch.busOn()
    
    def receive_frame(self):
        # Receive the message
        # TODO SOMETIMES WRONG ID 9_123. WHY? ./canmonitor 0 doesn't give those results
        # TODO: IF NOT MESSAGES it returns an error, is that ok?
        self.frame = self.ch.read(timeout=1000)
        # print(f'ID: {self.frame.id}')

    def send_frame(self, data=0):
        """
        Send all the target values from the agent
        """
        # TODO IF CAN'T SEND THE MESSAGE, ERROR MESSAGE BUT NOT EXCEPTION? or ok ERROR
        # TODO FOR LOOP AND SEND ALL THE TARGET VALUES, and relevant state values of the agent
        
        # Transmit the message
        ## Define frame
        frame = Frame(id_=123, data = [0,0,0,0,0,0] , dlc=6) # data = [1,2] , data=b'HELLO!'
        ## Send frame
        self.ch.write(frame)
        self.ch.writeSync(timeout=500) # Wait until sent, timeout of 500ms
        
        # print(f'FRAME SENT: {frame}')
        
    def turn_off(self):
        # Chip OFF
        self.ch.busOff()
        
        # Close channel
        self.ch.close()
    
    