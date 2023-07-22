from globals.globals import * # Global variables and constants, as if they were here
import can # python-can (Jetson)

class CanJetson():
    """
    Class that manages the CAN communications with the Nvidia Jetson Xavier.
    
    On Xavier, call first: enable_CAN_no_sudo_v2.sh
    TODO TRY TO CONFIGURE IN PYTHON instead of .sh
        INIT THAT CONFIGURE OR RUNS .SH FILE
    """
    
    def __init__(self):
        self.bus = can.interface.Bus(bustype='can0', channel=00, bitrate=1e6)  # bustype = 'can'
        self.logger.write_can_log(self.bus.channel_info)

        ## Implementamos la clase abstracta Listener (https://python-can.readthedocs.io/en/master/listeners.html)
        # De momento usamos BufferedReader, que es una implementación por defecto de la librería, con un buffer de mensajes. Si necesitamos que sea async podemos
        # usar AsyncBufferedReader.
        self.buffer = can.BufferedReader()
        # self.notifier = can.Notifier(self.bus, [self.buffer])
        # self.logger.write_can_log("CAN listener connected") # TODO LOGGER IN FUTURE
        
        
    def send_frame(self, data=0):
        """
        Send all the target values from the agent
        """
        # TODO IF CAN'T SEND THE MESSAGE, ERROR MESSAGE BUT NOT EXCEPTION? or ok ERROR
        # TODO FOR LOOP AND SEND ALL THE TARGET VALUES, and relevant state values of the agent
        
        # Transmit the message
        ## Define frame
        frame = can.Message(arbitration_id=id, data=data)
        ## Send frame
        try:
            self.bus.send(frame, timeout=0.005)
        except can.CanError as e:
            print(f"ERROR: Can't send CAN message (python-can, Jetson)\n {e}")
        
        # print(f'FRAME SENT: {frame}')
    def send_message(self, id, data):
        """
        
        Example: self.send_message(can_constants.STEER_ID['STEER_ID'], 6, CAN_IDS['STEER']['MSG_N'])
        """
        msg = can.Message(arbitration_id=id, data=data)
        
        try:
            self.bus.send(msg, timeout=0.005)
        except can.CanError as e:
            print("ERROR: Cannot send CAN message using Xavier")

