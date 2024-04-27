import os, sys, cv2, time
import numpy as np
import multiprocessing
from abc import ABC, abstractmethod

from trajectory_estimation.cone_processing import ConeProcessing #ConeProcessingNoWrapped
#import simple_pid
#from visualization_utils.logger import Logger
#from simple_pid import PID

class Agent():
    '''
    Main Agent class, with basic features. All other agents inherit from this one, adding specific functions.
    
    '''
    
    def __init__(self):
        self.cone_processing = ConeProcessing()
        self.speed_target = 5
        
        self.agent_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None
        self.agent_in_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None
        
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
    def _speed_control(self, cones, state, actuation, blues, yellows, oranges) -> None:

        # SPEED CONTROL - actuation ----- Take (target speed - current speed) -> PID
        actuation['acc'] = (self.speed_target - state['speed']) * 0.1
        brake_condition = (len(oranges) >= 6) and (oranges[0]['coords']['y'] < 1)
        
        if brake_condition: # If it sees enough ending orange cones, speed doesn't matter
            # Brake to finish
            actuation['steer'] = 0 # 1 left, -1 right, 0 neutral
            actuation['acc'] = 0.0
            actuation['brake'] = 1.0

            if(state['speed'] < 0.25): #Si se ha parado completamente, AS_Finished
                state['status'] = 'as_finished'
        elif actuation['acc'] < 0: # If negative acceleration
            # Brake instead
            actuation['brake'] = -actuation['acc']
            actuation['acc'] = 0
        else:
            # Normal speed regulation
            if state['speed'] < 5: # If going slow
                # Accelerate
                actuation['acc'] = 1.0
            else: # If going fast
                # Stop accelerating
                actuation['acc'] = 0.0
    
    def _steer_control(self, cones, state, actuation, blues, yellows, oranges) -> None:
        if (len(blues) > 0) and (len(yellows) > 0):
            # I assume they're sorted from closer to further
            center = (blues[0]['coords']['y'] + yellows[0]['coords']['y']) / 2 # positive means left
            # print(f'center:{center}')
            # Limit the value of 'center' between -1 and 1
            center = max(-1, min(1, center))
            actuation['steer'] = center * 0.5 # -1 left, 1 right, 0 neutral TODO HACER CON MAS SENTIDO
        elif len(blues) > 0:
            actuation['steer'] = -1 # Rotation in Z axis. - = right
        elif len(yellows) > 0:
            actuation['steer'] = 1 # Rotation in Z axis. + = left
        else:
            actuation['steer'] = 1.0 # Rotation in Z axis. + = left
    
    def get_target(self, cones, state, actuation) -> None:
        '''
        Update actuation, calculated from the cones and state.
        '''
        
        # Sort cones from closest to furthest
        def take_x(cone): return cone['coords']['x']
    
        blues = [cone for cone in cones if (cone['label'] == 'blue_cone')]
        blues.sort(key=take_x)
        
        yellows = [cone for cone in cones if (cone['label'] == 'yellow_cone')]
        yellows.sort(key=take_x)

        oranges = [cone for cone in cones if (cone['label'] == 'orange_cone')]
        oranges.sort(key=take_x)
        
        # In these cases, do nothing
        if state['status'] == 'as_finished' or state['status'] == 'as_emergency' or state['status'] == 'as_off':
            return
        
        self._speed_control(cones, state, actuation, blues, yellows, oranges)

        # STEER CONTROL
        self._steer_control(cones, state, actuation, blues, yellows, oranges)

        state['status'] = 'as_driving'
    
    @staticmethod
    def create(mode):
        """Factory method to create an Agent based on the mode.
        
        Agent selection
        - 0 -> Generic Agent: Runs continuously
        - 1 -> Acceleration
        - 2 -> Skidpad
        - 3 -> Autocross
        - 4 -> Trackdrive
        - 5 -> EBS Test
        - 6 -> Inspection
        - 7 -> Manual Drive
        """
        if (mode == 0): # Simple Agent
            from agent.simple_agent import SimpleAgent
            return SimpleAgent()
        elif (mode == 1): # Acceleration
            from agent.agent_acceleration_mission import Acceleration_Mission
            return Acceleration_Mission()
        elif (mode == 2): # Skidpad
            from agent.agent_skidpad_mission import Skidpad_Mission
            return Skidpad_Mission()
        elif (mode == 3): # Autocross
            from agent.agent import Agent
            return Agent()
        elif (mode == 4): # Trackdrive
            from agent.agent import Agent
            return Agent()
        # elif (mission == 5): # EBS Test
        #     from agent.agent_ebs_test import EBS_Test_Mission
        #     return EBS_Test_Mission(self.can0) 
        elif (mode == 6): # Inspection
            from agent.agent_inspection import Inspection_Mission
            return Inspection_Mission()
        # elif (mission == 7): # Manual Drive
        #     self.terminate() # La orin no necesitar hacer nada mas
        elif (mode == 13): # Skidpad
            from agent.agent_pablo import Agent_Pablo
            return Agent_Pablo()
        elif (mode == 14): # Skidpad
            from agent.agent_pablo_pre import Agent
            return Agent()
        else: # The default Agent is the class of which other agents inherit from
            raise Exception(f'ERROR: WRONG mission VALUE. Got {mode} but expected integer from 0 to 7')