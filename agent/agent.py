from trajectory_estimation.cone_processing import ConeProcessing #ConeProcessingNoWrapped
#import cv2
#import simple_pid
# import numpy as np
#from visualization_utils.logger import Logger
#from simple_pid import PID

from globals.globals import * # Global variables and constants, as if they were here

class Agent():
    '''
    Main Agent class, with basic features. All other agents inherit from this one, adding specific functions.
    
    '''
    
    def __init__(self):
        self.cone_processing = ConeProcessing()
        self.speed_target = 5
    """
    def valTrackbarsPID(self):
        '''
        returns the information from the trackbars
        '''
        kp = cv2.getTrackbarPos("Kp/1000", "Trackbars PID") / 1000
        ki = cv2.getTrackbarPos("Ki/1000", "Trackbars PID") / 1000
        kd = cv2.getTrackbarPos("Kd/1000", "Trackbars PID") / 1000
        throttle_kp = cv2.getTrackbarPos("Throttle Kp/100", "Trackbars PID") / 1000
        throttle_ki = cv2.getTrackbarPos("Throttle Ki/100", "Trackbars PID") / 1000
        throttle_kd = cv2.getTrackbarPos("Throttle Kd/100", "Trackbars PID") / 1000
        brake_kp = cv2.getTrackbarPos("Brake Kp/100", "Trackbars PID") / 1000
        brake_ki = cv2.getTrackbarPos("Brake Ki/1000", "Trackbars PID") / 1000
        brake_kd = cv2.getTrackbarPos("Brake Kd/1000", "Trackbars PID") / 1000

        # TODO TEST update self values instead of return
        self.pid_kp = kp
        self.pid_ki = ki
        self.pid_kd = kd
        self.pid_throttle_kp = throttle_kp
        self.pid_throttle_ki = throttle_ki
        self.pid_throttle_kd = throttle_kd
        self.pid_brake_kp = brake_kp
        self.pid_brake_ki = brake_ki
        self.pid_brake_kd = brake_kd

        return kp, ki, kd, throttle_kp, throttle_ki, throttle_kd, brake_kp, brake_ki, brake_kd
    """

    def get_target(self, cones, car_state, agent_act):
        '''
        Update agent_act, calculated from the cones and car_state.
        '''
        
        # SORT CONES FROM CLOSEST TO FURTHEST
        def take_x(cone): return cone['coords']['x']
        
        blues = [cone for cone in cones if (cone['label'] == 'blue_cone')]
        blues.sort(key=take_x)
        
        yellows = [cone for cone in cones if (cone['label'] == 'yellow_cone')]
        yellows.sort(key=take_x)

        orange = [cone for cone in cones if (cone['label'] == 'orange_cone')]
        orange.sort(key=take_x)

        # SPEED CONTROL - agent_act ----- Take (target speed - current speed) -> PID
        agent_act['acc'] = (self.speed_target - car_state['speed']) * 0.1
        brake_condition = (len(orange) >= 6) and (orange[0]['coords']['y'] < 1)
        
        # If negative acceleration, brake instead
        if agent_act['acc'] < 0:
            agent_act['brake'] = -agent_act['acc']
            agent_act['acc'] = 0
        


        if (car_state['speed'] < 5) and (not brake_condition):
            agent_act['acc'] = 1.0
        
        elif brake_condition: # da igual la velocidad, si ve conos naranjas
            agent_act['steer'] = 0 # 1 left, -1 right, 0 neutral
            agent_act['acc'] = 0.0
            agent_act['brake'] = 1.0

            if(car_state['speed'] < 0.25): #Si se ha parado completamente, AS_Finished
                return True
        else:
            agent_act['acc'] = 0.0
        
        # STEER CONTROL
        if (len(blues) > 0) and (len(yellows) > 0):
            # I assume they're sorted from closer to further
            center = (blues[0]['coords']['y'] + yellows[0]['coords']['y']) / 2 # positive means left
            # print(f'center:{center}')
            agent_act['steer'] = center * 0.5 # -1 left, 1 right, 0 neutral TODO HACER CON MAS SENTIDO
        elif len(blues) > 0:
            agent_act['steer'] = -1 # Rotation in Z axis. - = right
        elif len(yellows) > 0:
            agent_act['steer'] = 1 # Rotation in Z axis. + = left
        else:
            agent_act['steer'] = 1.0 # Rotation in Z axis. + = left

        return False