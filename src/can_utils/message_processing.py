class MessageProcessing():
    """
    Manage interpretation of CAN messages to update state variables
    """
    def __init__(self):
        
        # LINK ID TO EACH PARSE METHOD FOR CAN MESSAGES. FOR NOW IN __INIT__ SO THE METHODS ARE ACCESSIBLE.
        self.CAN_ID_TO_METHODS = {
            0x300: self.parse_senfl_imu,
            0x301: self.parse_senfl_sig, # Various signals. Speed and wheel turns
            0x302: self.parse_senfl_state,
            
            0x303: self.parse_senfr_imu,
            0x304: self.parse_senfr_sig,
            0x305: self.parse_senfr_state,
            
            0x320: self.parse_traj_act,
            0x321: self.parse_traj_gps,
            0x322: self.parse_traj_imu,
            0x323: self.parse_traj_state,
            
            # Steer ... TODO in the future
            
            # ASSIS
            0x350: self.parse_assis_c,
            0x351: self.parse_assis_r,
            0x352: self.parse_assis_l,
            
            0x360: self.parse_asb_analog,
            0x361: self.parse_asb_signals,
            0x362: self.parse_asb_state,
            
            0x201: self.parse_arduino,
        }
        
        
    def can_message_to_state(frame, car_state):
        """
        - Takes a CAN message, and the car_state dictionary
        - Returns the updated dictionary according to the CAN message
        
        """
        
        # PROCESS FRAME - https://pycanlib.readthedocs.io/en/latest/canlib/canframes.html
        id = frame.id
        data = [byte for byte in frame.data]
        dlc = frame.dlc
        timestamp = frame.timestamp
        # flags = frame.flags
        
        # Run the method
        car_state = self.run_method(id, car_state)
        
        
        # TODO Get id etc, then look how to compare with ids from can_constants
        
        return car_state
    
    def run_method(self, id, car_state):
        """
        - Runs a method according to the id and MessageProcessing dictionary with id and method names.
        - If the id is not in the dictionary, prints an ERROR TODO USE LOGGER
        
        Each CAN message requires different processing according to its id.
        id is a natural integer
        """
        # If the id is somewhere in the dictionary
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