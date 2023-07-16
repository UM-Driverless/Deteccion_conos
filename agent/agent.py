from trajectory_estimation.cone_processing import ConeProcessing #ConeProcessingNoWrapped
#import cv2
#import simple_pid
#import numpy as np
#from visualization_utils.logger import Logger
#from simple_pid import PID

from globals.globals import * # Global variables and constants, as if they were here

# TODO SIMPLE AGENT THAT INHERITS FROM GENERAL

class Agent():
    '''
    Main Agent class, with all the common actions between agents.
    Each different test will inherit from this class and implement the specific actions for the test.
    '''
    def __init__(self):
        self.cone_processing = ConeProcessing()

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

    def act_sim(self, cones, sim_client2, simulator_car_controls):
        '''
        Control the simulator. Runs _get_target with more stuff. Higher level function

        1. Read speed from simulator
        2. Calculate next move with _get_target()
        3. Send the next move to the simulator
        '''
        # Read Simulator
        sim_state = sim_client2.getCarState()
        car_state['speed'] = sim_state.speed
        
        # Calculate agent_target
        self._get_target(cones)
        
        # Update Simulator
        simulator_car_controls.steering = agent_target['steer']
        simulator_car_controls.throttle = agent_target['acc']
        simulator_car_controls.brake = agent_target['brake']

        sim_client2.setCarControls(simulator_car_controls)

    def act(self, cones):
        '''
        
        '''
        self._get_target(cones)
        
    def _get_target(self, cones):
        '''
        Update agent_target, calculated from the cones and car_state.
        '''
        
        # SORT CONES FROM CLOSEST TO FURTHEST
        def take_x(cone): return cone['coords']['x']
        
        blues = [cone for cone in cones if (cone['label'] == 'blue_cone')]
        blues.sort(key=take_x)
        
        yellows = [cone for cone in cones if (cone['label'] == 'yellow_cone')]
        yellows.sort(key=take_x)

        # SPEED CONTROL - agent_target
        if (car_state['speed'] < 5):
            agent_target['acc'] = 1.0
        else:
            agent_target['acc'] = 0.0
        
        # STEER CONTROL
        if (len(blues) > 0) and (len(yellows) > 0):
            # I assume they're sorted from closer to further
            center = (blues[0]['coords']['y'] + yellows[0]['coords']['y']) / 2 # positive means left
            # print(f'center:{center}')
            agent_target['steer'] = (-center) * 0.5 # -1 left, 1 right, 0 neutral TODO HACER CON MAS SENTIDO
        elif len(blues) > 0:
            agent_target['steer'] = 1 # -1 left, 1 right, 0 neutral
        elif len(yellows) > 0:
            agent_target['steer'] = -1 # -1 left, 1 right, 0 neutral
        else:
            agent_target['steer'] = -1 # left to see some cones or go in circles
