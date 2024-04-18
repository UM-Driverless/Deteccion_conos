import os, sys, time, math, cv2, yaml
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt # For representation of time consumed

from visualization_utils.logger import Logger
from cone_detection.yolo_detector import ConeDetector

from camera import Camera
from comm import CarCommunicator
from agent.agent import Agent

class Car:
    # Possible config constants here. In the future in yaml file
    '''
    1. get_data -> _get_image, _get_sensors
    2. calculate_actuation -> call the right agent
    3. send_actuation -> send CAN messages
    
    '''
    def __init__(self, CONFIG_FILEPATH = "config.yaml"):
        with open(CONFIG_FILEPATH, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.state = {
            'status': 'default', # as_off, as_ready, as_driving, as_emergency, as_finished TODO GOOD STATE MACHINE OR RECEIVE FROM PMC. THIS DOES SOME CHANGES. THE CURRENT IDEA IS TO HAVE THE STATE MACHINE IN PMC, TRANSMIT TO ORIN, WHEN ORIN DETECTS END OF TEST, SEND TO PMC SO THAT IT CHANGES THE STATE MACHINE.
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

        self.SRC_DIR = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.SRC_DIR)
        
        self.loop_counter: int = 0
        # Neural net detections [{bounding boxes},{labels}] = [ [[[x1,y1],[x2,y2]], ...], [[{class name}, {confidence}], ...] ]
        self.detections = []
        
        if self.config['LOGGER']:
            self.logger = Logger()
            # Log config
            msg = "CONFIG: "
            for key, value in self.config.items():
                msg += f"{key}: {value}, "
            self.logger.write(msg, msg)
        
        self.camera = Camera.create(self.config['CAMERA_MODE'])
        self.camera.start()
        self.comm = CarCommunicator.create(self.config['COMM_MODE'])
        
        self.cones = []
        self.agent = Agent.create(self.config['AGENT_MODE'])
        
        self.detector = ConeDetector(checkpoint_path=self.config['WEIGHTS_PATH'])
    
    def get_data(self):
        '''
        Main method that gets data from sensors, the image etc, and writes it to the car instance variables.
        All data should be synced.
        
        ---
        1. Get image from camera
        Takes the last image taken by the camera from cam_queue, and writes it into self.image.
        
        When the image is taken from the queue, the next one will be added as fast as possible in a separate thread.
        
        If the image still isn't there, it will wait, slowing down the code. This would be the bottleneck then.
        '''
        self.image = self.camera.get_image()
        self.comm.receive_state(self.state)
    
    def _can_send_thread(self):
        print(f'Starting CAN receive thread...')
        
        while True:
            self.can_receive.receive_frame() # self.can_receive.frame updated
            # print(f'FRAME RECEIVED: {self.can_receive.frame}')
            # global car_state
            car_state_local = self.can_receive.new_state(self.state)
            # print(car_state_local)
            self.can_queue.put(car_state_local)
    
    def stop(self):
        self.camera.stop()
        self.actuation = {
            'acc': 0., # Acceleration. From -1.0 to 1.0.
            'steer': 0., # -1.0 (right) to 1.0 (left)
            'throttle': 0., # float in [0., 1.)
            'brake': 0., # float in [0., 1.)
        }
        