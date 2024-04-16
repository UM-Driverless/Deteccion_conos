import os, sys, time, math, cv2
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt # For representation of time consumed

from globals.globals import * # Global variables and constants, as if they were here
from visualization_utils.logger import Logger
from cone_detection.yolo_detector import ConeDetector

from camera import Camera
from camera import ImageFileCamera
from camera import VideoFileCamera

class Car:
    # Possible config constants here. In the future in yaml file
    '''
    1. get_data -> _get_image, _get_sensors
    2. calculate_actuation -> call the right agent
    3. send_actuation -> send CAN messages
    
    '''
    def __init__(self):
        
        self.state = {
            'speed': -1., # m/s, can be calculated from speed_senfl mixed with other sensors.
            'rpm': -1.,
            'speed_senfl': -1., # speed according to SEN front left sensor
            'fps': -1.0,
            'orientation_y_rad': 0., # By default consider 0.
        }
        
        self.actuation = {
            # 'acc_normalized': 0., # Acceleration. From -1.0 to 1.0. TODO Then translates into throttle and brake
            'acc': 0., # Acceleration of the car, 0.0 to 1.0
            'steer': 0., # -1.0 (right) to 1.0 (left)
            'throttle': 0., # float in [0., 1.)
            'brake': 0., # float in [0., 1.)
        }

        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.ROOT_DIR)
        
        self.loop_counter: int = 0
        # Neural net detections [{bounding boxes},{labels}] = [ [[[x1,y1],[x2,y2]], ...], [[{class name}, {confidence}], ...] ]
        self.detections = []
        
        self.logger = Logger()
        
        self._init_cam()
        self._init_can()
        self._init_agent()
        
        self.detector = ConeDetector(checkpoint_path=WEIGHTS_PATH)
        self.cones = []

    def _init_cam(self):
        '''
        Init the camera thread according to the CAMERA_MODE selected.
        '''
        self.cam_queue  = multiprocessing.Queue(maxsize=1) #block=True, timeout=None
        
        self.camera = Camera.create(CAMERA_MODE, self.cam_queue)
        self.camera.start()
    
    def get_data(self):
        '''
        Main method that gets data from sensors, the image etc, and writes it to the car instance variables.
        All data should be synced.
        '''
        self._get_image()
        
        if (CAMERA_MODE == 4):
            # Take control of the simulator
            self.sim_client2.enableApiControl(True)
            
            # Read speed from simulator
            sim_state = self.sim_client2.getCarState()
            self.state['speed'] = sim_state.speed
    
    def _get_image(self):
        '''
        Takes the last image taken by the camera from cam_queue, and writes it into self.image.
        
        When the image is taken from the queue, the next one will be added as fast as possible in a separate thread.
        
        If the image still isn't there, it will wait, slowing down the code. This would be the bottleneck then.
        '''
        self.image = self.cam_queue.get(timeout=4)
    
    def _init_can(self):
        '''
        Initialize the CAN communications according the the CAN_MODE settings.
        '''
        if (CAN_MODE == 1):
            from can_utils.can_utils import CAN
        elif (CAN_MODE == 2):
            from can_utils.can_kvaser import CanKvaser
        
        if (CAN_MODE == 1):
            # CAN with Jetson
            self.can0 = CAN()
            print('CAN (python-can, socketcan, Jetson) initialized')

        elif (CAN_MODE == 2):
            # CAN with Kvaser

            self.can_receive = CanKvaser()
            self.can_send = CanKvaser()
            self.can_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None TODO probably bad parameters, increase maxsize etc.
            
            self.can_send_worker = multiprocessing.Process(target=self._can_send_thread, args=(), daemon=False)
            self.can_send_worker.start()
            
            print('CAN (Kvaser) initialized')
    
    def _can_send_thread(self):
        print(f'Starting CAN receive thread...')
        
        while True:
            self.can_receive.receive_frame() # self.can_receive.frame updated
            # print(f'FRAME RECEIVED: {self.can_receive.frame}')
            # global car_state
            car_state_local = self.can_receive.new_state(self.state)
            # print(car_state_local)
            self.can_queue.put(car_state_local)


    def _init_agent(self):
        '''
        Initialize the agent thread according to the MISSION_SELECTED settings
        '''
        
        ## Agent selection - 0 -> Generic Agent: Runs continuously, 1 -> Acceleration, 2 -> Skidpad, 3 -> Autocross, 4 -> Trackdrive, 5 -> EBS Test, 6 -> Inspection, 7 -> Manual Drive
        if (MISSION_SELECTED == 0): # Generic, we dont use it really
            from agent.agent import Agent
            self.agent = Agent()
        elif (MISSION_SELECTED == 1): # Acceleration
            from agent.agent_acceleration_mission import Acceleration_Mission
            self.agent = Acceleration_Mission()
        elif (MISSION_SELECTED == 2): # Skidpad
            from agent.agent_skidpad_mission import Skidpad_Mission
            self.agent = Skidpad_Mission()
        elif (MISSION_SELECTED == 3): # Autocross
            from agent.agent import Agent
            self.agent = Agent()
        elif (MISSION_SELECTED == 4): # Trackdrive
            from agent.agent import Agent
            self.agent = Agent()
        elif (MISSION_SELECTED == 5): # EBS Test
            from agent.agent_ebs_test import EBS_Test_Mission
            self.agent = EBS_Test_Mission(self.can0) 
        elif (MISSION_SELECTED == 6): # Inspection
            from agent.agent_inspection import Inspection_Mission
            self.agent = Inspection_Mission()
        elif (MISSION_SELECTED == 7): # Manual Drive
            self.terminate() # La orin no necesitar hacer nada mas
        elif (MISSION_SELECTED == 13): # Skidpad
            from agent.agent_pablo import Agent_Pablo
            self.agent = Agent_Pablo()
        elif (MISSION_SELECTED == 14): # Skidpad
            from agent.agent_pablo_pre import Agent
            self.agent = Agent()
        else: # The default Agent is the class of which other agents inherit from
            raise Exception(f'ERROR: WRONG MISSION_SELECTED VALUE. Got {MISSION_SELECTED} but expected integer from 0 to 7')

        self.agent_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None
        self.agent_in_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None


    def calculate_actuation(self):
        '''
        Method that calculates the trajectory and writes it to self.actuation
        '''
        
        return self.agent.get_target(self.cones, self.state, self.actuation)

    def send_actuation(self):
        '''
        Sends the self.actuation values to move the car.
        '''
        if (CAMERA_MODE == 4): # SIMULATOR AUTONOMOUS MODE
            
            # Send to Simulator
            self.simulator_car_controls.steering = -self.actuation['steer'] # + rotation is left for us, right for simulator
            self.simulator_car_controls.throttle = self.actuation['acc']
            self.simulator_car_controls.brake = self.actuation['brake']

            self.sim_client2.setCarControls(self.simulator_car_controls)
        
        if (CAN_MODE == 1):
            self.can0.send_action_msg(self.actuation)
    
    def terminate(self):
        if (CAMERA_MODE == 2): # Webcam
            cv2.destroyAllWindows()
        if (CAMERA_MODE == 3): # ZED
            # Close cameras
            self.zed.close()
            cv2.destroyAllWindows()
        if (CAMERA_MODE == 4): # Simulator autonomous
            # Give sim control back
            self.sim_client2.enableApiControl(False) # Allows mouse control, only API with this code

        self.actuation = {
            'acc': 0., # Acceleration. From -1.0 to 1.0.
            'steer': 0., # -1.0 (right) to 1.0 (left)
            'throttle': 0., # float in [0., 1.)
            'brake': 0., # float in [0., 1.)
        }
        