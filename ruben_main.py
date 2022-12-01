
# MAIN script to execute all the others. Shall contain the main classes, top level functions etc.

# TODO
# add requirements for our specific extra libraries (cv2, pythoncan, opencv-python, ...)
#
#

# LINKS
# https://github.com/UM-Driverless/Deteccion_conos/tree/Test_Portatil
#
#


# Constants to define what to do
CAN_MODE = 0 # 0 -> CAN OFF, default values to test without CAN, 1 -> KVaser, 2 -> Arduino
CAMERA_MODE = 1 # 0 -> Webcam, 1 -> Read video file (VIDEO_FILE_NAME required), 2 -> ZED
VIDEO_FILE_NAME = 'test_video.mp4' # Only used if CAMERA_MODE == 1
VISUALIZE = 1

WEIGHTS_PATH = 'yolov5/weights/yolov5_models/800.pt'
#WEIGHTS_PATH = 'yolov5/weights/yolov5_models/TensorRT/240.engine' # TODO MAKE IT WORK with tensorrt weights


# IMPORTS
from connection_utils.car_comunication import ConnectionManager
from controller_agent.agent import AgentAccelerationYolo as AgentAcceleration
from cone_detection.yolo_detector import ConeDetector
from visualization_utils.visualizer_yolo_det import Visualizer
from visualization_utils.logger import Logger

import os
import time
import numpy as np
import matplotlib.pyplot as plt # For representation of time consumed
# TODO add dependency sudo apt-get install python3-tk

## Camera libraries
import cv2 # Webcam
#import pyzed.sl as sl # ZED.

# TODO ADD ZED dependencies to requirements.txt: python -m pip install cython numpy opencv-python pyopengl

if __name__ == '__main__':
    # SETUP CAMERA
    if (CAMERA_MODE == 0):
        # WEBCAM
        
        # init.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
        # init.coordinate_units = sl.UNIT.METER
        # init.depth_minimum_distance = 0.15

        # runtime.sensing_mode = sl.SENSING_MODE.FILL
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FPS, 30)
    elif (CAMERA_MODE == 1):
        # Read video file
        cam = cv2.VideoCapture(VIDEO_FILE_NAME)
        
    elif (CAMERA_MODE == 2):
        # ZED
        # TODO CHECK if this works with zed
        
        """
        cam = sl.Camera()
        cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 0)
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.camera_fps = 30
        # init.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
        # init.coordinate_units = sl.UNIT.METER
        # init.depth_minimum_distance = 0.15

        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit(1)

        runtime = sl.RuntimeParameters()
        # runtime.sensing_mode = sl.SENSING_MODE.FILL
        """

    # INITIALIZE things
    ## Logger
    logger_path = os.path.join(os.getcwd(), "logs")
    init_message = "actuator_zed_testing.py"
    logger = Logger(logger_path, init_message)
    
    ## Cone detector
    detector = ConeDetector(checkpoint_path=WEIGHTS_PATH, logger=logger)
    
    ## Connections
    #connect_mng = ConnectionManager(logger=logger)

    ## Agent
    agent = AgentAcceleration(logger=logger, target_speed=60.)

    ## Data visualization
    visualizer = Visualizer()


    # READ TIMES
    recorded_times = [0]*10
    start_time = -1
    fps = -1
    loop_counter = 0

    # Main loop
    try:
        while True:
            start_time = time.time()

            # ASK DATA (To the car sensors or the simulator)
            #in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(VISUALIZE=1)

            result, image = cam.read() # TODO check if result == true?
            recorded_times[0] += time.time() - start_time
            np.array(image)

            recorded_times[1] += time.time() - start_time
            # cv2.imshow("img", image)
            # cv2.waitKey(1)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            recorded_times[2] += time.time() - start_time

            # Detect cones
            detections, cone_centers = detector.detect_cones(image, get_centers=True)
            recorded_times[3] += time.time() - start_time
            # Actions:
            # 1 -> steer
            # 2 -> throttle
            # 3 -> brake
            # 4 -> clutch
            # 5 -> upgear
            # 6 -> downgear

            # Seleccionar acciones
            [throttle, brake, steer, clutch, upgear, downgear, gear], data = agent.get_action(detections=detections,
                                                                                                speed=0,
                                                                                                gear=0,
                                                                                                rpm=0,
                                                                                                cone_centers=cone_centers,
                                                                                                image=image)
            recorded_times[4] += time.time() - start_time
            # resize actions
            # throttle *= 0.8
            # brake *= 0.8
            # steer *= 0.8
            # clutch *= 0.8

            # Enviar acciones
            '''connect_mng.send_actions(throttle=throttle,
                                        brake=brake,
                                        steer=steer,
                                        clutch=clutch,
                                        upgear=upgear,
                                        downgear=downgear)'''
            
            

            if VISUALIZE == 1:
                cenital_map = [data[1], data[2], data[-1]]
                in_speed = 0
                in_gear = 0
                in_rpm = 0

                visualizer.visualize([image, detections, cone_centers,cenital_map, in_speed],
                                     [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm], fps,
                                     save_frames=False)
            
            recorded_times[5] += time.time() - start_time

            #print(f'TIMES: {recorded_times[0]}, {recorded_times[1]}, {recorded_times[2]}, {recorded_times[3]}, {recorded_times[4]}')
            fps = 1/(time.time() - start_time)
            print(f'FPS: {fps}')
            loop_counter += 1

    finally:
        # When main loop stops,

        # PRINT TIMES
        print(f'\n')
        print(f'LOOPS: {loop_counter}')
        #print(f'INTEGRATED TIMES: {recorded_times[0]}, {recorded_times[1]}, {recorded_times[2]}, {recorded_times[3]}, {recorded_times[4]}')
        print(f'AVERAGE TIMES: {recorded_times[0]/loop_counter}, {recorded_times[1]/loop_counter}, {recorded_times[2]/loop_counter}, {recorded_times[3]/loop_counter}, {recorded_times[4]/loop_counter}')

        fig = plt.figure(figsize=(12, 4))
        plt.bar(['cam.read()','Make np.array()','cv2.cvtColor()','detect_cones()','agent.get_action()'],[t/loop_counter for t in recorded_times[0:5]])
        plt.ylabel("Average time taken [s]")
        #plt.title("Title")
        #plt.show()
        plt.savefig("logs/times.png")

        throttle = 0
        brake = 0
        steer = 0
        clutch = 0
        upgear = 0
        downgear = 0

        #visualizer.close_windows() # TODO 'Visualizer' object has no attribute 'close_windows'




