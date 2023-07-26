import os
print(f'Current working directory: {os.getcwd()}') # The terminal should be in this directory
import sys
print(f'Python version: {sys.version}')

import time
import math
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt # For representation of time consumed
import cv2

from globals.globals import * # Global variables and constants, as if they were here
from visualization_utils.logger import Logger
from cone_detection.yolo_detector import ConeDetector

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
            'acc_normalized': 0., # Acceleration. From -1.0 to 1.0. TODO Then translates into throttle and brake
            'steer': 0., # -1.0 (right) to 1.0 (left)
            'throttle': 0., # float in [0., 1.)
            'brake': 0., # float in [0., 1.)
        }

        self.loop_counter: int = 0
        # Neural net detections [{bounding boxes},{labels}] = [ [[[x1,y1],[x2,y2]], ...], [[{class name}, {confidence}], ...] ]
        self.detections = []
        
        self.logger = Logger("Logger initialized")
        
        
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
        
        if (CAMERA_MODE == 0):   self.cam_worker = multiprocessing.Process(target=self._read_image_file,   args=(), daemon=False)
        elif (CAMERA_MODE == 1): self.cam_worker = multiprocessing.Process(target=self._read_image_video,  args=(), daemon=False)
        elif (CAMERA_MODE == 2): self.cam_worker = multiprocessing.Process(target=self._read_image_webcam, args=(), daemon=False)
        elif (CAMERA_MODE == 3): self.cam_worker = multiprocessing.Process(target=self._read_image_zed,    args=(), daemon=False)
        elif (CAMERA_MODE == 4): self.cam_worker = multiprocessing.Process(target=self._read_image_simulator, args=(), daemon=False)
        elif (CAMERA_MODE == 5): self.cam_worker = multiprocessing.Process(target=self._read_image_simulator, args=(), daemon=False)
        
        if (CAMERA_MODE == 3):
            ### ZED_IMU
            import pyzed.sl as sl
            self.zed = sl.Camera()
            self.zed_sensors = sl.SensorsData()
            zed_params = sl.InitParameters()
            zed_params.coordinate_units = sl.UNIT.METER
            zed_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD


        if (CAMERA_MODE == 4):
            # With https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator cloned in the home directory:
            fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python") # os.getcwd()
            sys.path.insert(0, fsds_lib_path)
            print(f'FSDS simulator path: {fsds_lib_path}')
            import fsds # TODO why not recognized when debugging
            
            # connect to the simulator
            self.sim_client1 = fsds.FSDSClient() # To get the image
            self.sim_client2 = fsds.FSDSClient() # To control the car
            # TODO TRY THIRD CLIENT SO THE SIMULATOR AND MOUSE CAN WORK TOGETHER

            # Check network connection, exit if not connected
            self.sim_client1.confirmConnection()
            self.sim_client2.confirmConnection()
            
            # Control the Car
            self.sim_client2.enableApiControl(True) # Disconnects mouse control, only API with this code
            self.simulator_car_controls = fsds.CarControls()

        elif (CAMERA_MODE == 5):
            # With https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator cloned in the home directory:
            fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
            sys.path.insert(0, fsds_lib_path)
            print(f'FSDS simulator path: {fsds_lib_path}')
            import fsds # TODO why not recognized when debugging
            
            # connect to the simulator
            self.sim_client1 = fsds.FSDSClient() # To get the image
            self.sim_client2 = fsds.FSDSClient() # To control the car

            # Check network connection, exit if not connected
            self.sim_client1.confirmConnection()
            self.sim_client2.confirmConnection()
            
            # No control. Give to keyboard
            self.sim_client2.enableApiControl(False)
        
        self.cam_worker.start()
    
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
    
    def _read_image_file(self):
        '''
        Reads an image file
        '''
        
        print(f'Starting _read_image_file thread...')
        
        while True:
            image = cv2.imread(IMAGE_FILE_NAME)
            
            self.cam_queue.put(image)
            
            # cv2.imshow('im',image)
            # cv2.waitKey(1)
    
    def _read_image_video(self):
        '''
        Reads a video file
        '''
        import cv2
        
        print(f'Starting read_image_video thread...')

        cam = cv2.VideoCapture(VIDEO_FILE_NAME)
        
        # SETTINGS
        # cam.set(cv2.CAP_PROP_FPS, 60)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640) #1280 640 default
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
        # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase
        
        if (cam.isOpened() == False): 
            print('Error opening video file')
        
        while True:
            # recorded_times_0 = time.time()
            result, image = cam.read() # TODO also check CHECK cam.isOpened()?
            while result == False:
                result, image = cam.read()
            
            # print(f'isOpened: {cam.isOpened()}')
            # cv2.imshow('image',image)
            # cv2.waitKey(10)
            
            # recorded_times_1 = time.time()
            
            self.cam_queue.put(image)
            # print(f'Video read time: {recorded_times_1-recorded_times_0}')

    def _read_image_webcam(self):
        '''
        Reads the webcam
        It usually takes about 35e-3 s to read an image, but in parallel it doesn't matter.
        '''
        print(f'Starting _read_image_webcam thread...')

        cam = cv2.VideoCapture(CAM_INDEX)
        
        # SETTINGS
        # cam.set(cv2.CAP_PROP_FPS, 60)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640) #1280 640 default
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
        # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase
        
        if (cam.isOpened() == False):
            print('Error opening webcam')
        
        while True:
            # recorded_times_0 = time.time()
            
            # Read image from webcam
            # TODO also check CHECK cam.isOpened()?
            # It's 3 times faster if there are cones being detected. Nothing to do with visualize.
            result, image = cam.read()
            while result == False:
                result, image = cam.read()
            
            # recorded_times_1 = time.time()
            
            if FLIP_IMAGE:
                image = cv2.flip(image, flipCode=1) # For testing purposes
            
            self.cam_queue.put(image)
            # print(f'Webcam read time: {recorded_times_1 - recorded_times_0}')

    def _read_image_simulator(self):    
        import fsds
        while True:
            [img] = self.sim_client1.simGetImages([fsds.ImageRequest(camera_name = 'cam1', image_type = fsds.ImageType.Scene, pixels_as_float = False, compress = False)], vehicle_name = 'FSCar')
            img_buffer = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
            image = img_buffer.reshape(img.height, img.width, 3)
            self.cam_queue.put(image)
    
    def _read_image_zed(self):
        '''
        Read the ZED camera - https://www.stereolabs.com/docs/video/camera-controls/
        '''
        print(f'Starting read_image_zed (opencv) thread...')

        cam = cv2.VideoCapture(CAM_INDEX)
        
        # SETTINGS
        # cam.set(cv2.CAP_PROP_FPS, 60)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640*2) #1280 640 default
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
        # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase
        
        try:
            if (cam.isOpened() == False): 
                # print("ErrStarting read_image_zed (opencv) threador opening webcam")
                raise Exception("ErrStarting read_image_zed (opencv) threador opening webcam")
        except Exception as e:
            print(e)
            self.terminate()
            return

        ######## ZED SENSORS
        # OPEN THE CAMERA
        # status = zed.open(zed_params)
        # while status != sl.ERROR_CODE.SUCCESS:
        #     print(f'ZED ERROR: {status}')
        #     status = zed.open(zed_params)
        # print('SUCESS, ZED opened')
        
        
        
        while True:
            # recorded_times_0 = time.time()
            
            # Read image from webcam
            # TODO also check CHECK cam.isOpened()?
            # It's 3 times faster if there are cones being detected. Nothing to do with visualize.
            result, image = cam.read()
            while result == False:
                result, image = cam.read()
            
            # recorded_times_1 = time.time()
            
            # cv2.imshow('image',image)
            # cv2.waitKey(1)
            
            image = cv2.resize(image, (640*2,640), interpolation=cv2.INTER_AREA)
            image = np.array(image)[:,0:640,:]
            
            if (FLIP_IMAGE):
                image = cv2.flip(image, flipCode=1) # For testing purposes
            
            self.cam_queue.put(image)
            # print(f'Webcam read time: {recorded_times_1 - recorded_times_0}')
            
            ########## ZED SENSORS
            # zed.get_sensors_data(sensors,sl.TIME_REFERENCE.IMAGE)
            # quaternions = sensors.get_imu_data().get_pose().get_orientation().get()
            # car_state['orientation_y_rad'] = math.atan2(2*quaternions[1]*quaternions[3] - 2*quaternions[0]*quaternions[2], 1 - 2*quaternions[1]**2 - 2 * quaternions[2]**2)
        
        '''
        import pyzed.sl as sl
        
        print(f'Starting read_image_zed thread...', end='')
        zed = sl.Camera()
        print(f' (ZED SDK version: {zed.get_sdk_version()})')
        
        # Init parameters: https://www.stereolabs.com/docs/api/structsl_1_1InitParameters.html
        zed_params = sl.InitParameters()
        sensors = sl.SensorsData() # TODO MOVE TO THREAD FOR HIGHER FREQUENCY? LINKED TO IMAGE SEEMS OK.
        # zed_params.camera_fps = 100 # Not necessary. By default does max fps
        
        # OPEN THE CAMERA
        status = zed.open(zed_params)
        while status != sl.ERROR_CODE.SUCCESS:
            print(f'ZED ERROR: {status}')
            status = zed.open(zed_params)
        print('SUCESS, ZED opened')
        
        # Camera settings
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 100) # We don't want blurry photos, we don't care about noise. The exposure time will still be adjusted automatically to compensate lighting conditions
        zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 8) # Maximum so it recognizes the color of the cones better. 0 to 8
        # cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 0) # Doesn't seem to make much difference. 0 to 8
        #cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1) # We have fixed gain, so this is automatic. % of camera framerate (period)
        
        # RESOLUTION: HD1080 (3840x1080), HD720 (1280x720), VGA (VGA=1344x376)
        # yolov5 uses 640x640. VGA is much faster, up to 100Hz
        zed_params.camera_resolution = sl.RESOLUTION.HD720
        zed_params.coordinate_units = sl.UNIT.METER
        zed_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        print(f'ZED ORIENTATION VALUE: {sensors.get_imu_data().get_pose().get_orientation().get()}')
        # car_state['orientation_y'] = sensors.get_imu_data().get_pose().get_orientation().get()
        #zed_params.sdk_gpu_id = -1 # Select which GPU to use. By default (-1) chooses most powerful NVidia
        
        runtime = sl.RuntimeParameters()
        runtime.enable_depth = False # Deactivates depth map calculation. We don't need it.
        zed_params.depth_mode = sl.DEPTH_MODE.NONE
        
        # Create an RGBA sl.Mat object
        mat_img = sl.Mat()
        
        while True:
            # Read ZED camera
            if (zed.grab(runtime) == sl.ERROR_CODE.SUCCESS): # Grab gets the new frame
                print('.')
                # recorded_times_0 = time.time()
                
                zed.retrieve_image(mat_img, sl.VIEW.LEFT) # Retrieve receives it and lets choose views and colormodes
                image = mat_img.get_data() # Creates np.array()
                self.cam_queue.put(image)
                
                # recorded_times_1 = time.time()
                # print(f'ZED read time: {recorded_times_1-recorded_times_0}')
                zed.get_sensors_data(sensors,sl.TIME_REFERENCE.IMAGE)
                quaternions = sensors.get_imu_data().get_pose().get_orientation().get()
                car_state['orientation_y_rad'] = math.atan2(2*quaternions[1]*quaternions[3] - 2*quaternions[0]*quaternions[2], 1 - 2*quaternions[1]**2 - 2 * quaternions[2]**2)
        '''
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
            car_state_local = self.can_receive.new_state(car_state)
            # print(car_state_local)
            self.can_queue.put(car_state_local)


    def _init_agent(self):
        '''
        Initialize the agent thread according to the MISSION_SELECTED settings
        '''
        
        ## Agent selection
        if (MISSION_SELECTED == 0): # Generic
            from agent.agent import Agent
            self.agent = Agent()
        elif (MISSION_SELECTED == 1): # Acceleration
            from agent.agent_acceleration_mission import Acceleration_Mission
            self.agent = Acceleration_Mission()
        elif (MISSION_SELECTED == 2): # Skidpad
            from agent.agent_skidpad_mission import Skidpad_Mission
            self.agent = Skidpad_Mission()
        else: # The default Agent is the class of which other agents inherit from
            raise Exception(f'ERROR: WRONG MISSION_SELECTED VALUE. Got {MISSION_SELECTED} but expected integer from 0 to 2')

        self.agent_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None
        self.agent_in_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None


    def calculate_actuation(self):
        '''
        Method that calculates the trajectory and writes it to self.actuation
        '''
        
        self.agent.get_target(self.cones, self.state, self.actuation)


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
        # Give sim control back
        if (CAMERA_MODE == 3):
            self.zed.close()
        if (CAMERA_MODE == 4):
            self.sim_client2.enableApiControl(False) # Allows mouse control, only API with this code

        self.actuation = {
            'acc': 0., # Acceleration. From -1.0 to 1.0.
            'steer': 0., # -1.0 (right) to 1.0 (left)
            'throttle': 0., # float in [0., 1.)
            'brake': 0., # float in [0., 1.)
        }
        