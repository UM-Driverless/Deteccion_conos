# TODO is this a module, a library, package or what? should we create an init file?

from canlib import canlib, Frame
from globals.globals import * # Global variables and constants, as if they were here
import time
import codecs

class Can_communication:
    """
    Class that manages the CAN communications. Send and receive messages using KVaser for now.
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
    
    def __init__(self):
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
        
        # Open the CAN Channel
        self.ch = canlib.openChannel(channel=0,flags=0,bitrate=canlib.Bitrate.BITRATE_1M)
        
        # Acceptance Filters - Hardware Level filters to reduce load - https://pycanlib.readthedocs.io/en/latest/canlib/sendandreceive.html#reading-messages
        # If the bits where the mask = 1 match with the code, the message is accepted.
        # ch.canAccept(0b1111,flag=canlib.AcceptFilterFlag.SET_MASK_STD)
        # ch.canAccept(0b0001,flag=canlib.AcceptFilterFlag.SET_CODE_STD)        
        
        # Set the CAN bus driver type (Silent, Normal...)
        self.ch.setBusOutputControl(canlib.Driver.NORMAL)
        
        # Activate the CAN chip
        self.ch.busOn()
        
    def receive_frame(self):
        # Receive the message
        frame = self.ch.read(timeout=1000) # TODO: IF NOT MESSAGES, OK? Now it returns an error
        
        
        # TODO PROCESS FRAME - https://pycanlib.readthedocs.io/en/latest/canlib/canframes.html
        # data = codecs.decode(frame.data,'utf-8')
        data = [a for a in frame.data]
        
        return data

            
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
            
            # TODO RETURN VARIABLES TO THREAD, TO SEND THEM THROUGH QUEUE, TO UPDATE THEM IN MAIN LOOP. TRY TO UPDATE LOCAL VARIABLES UNTIL REQUIRED TO SEND
            
            
    def send_frame(self, data=0):
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