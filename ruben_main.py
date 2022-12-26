
# MAIN script to execute all the others. Shall contain the main classes, top level functions etc.

"""
# LINKS
https://github.com/UM-Driverless/Deteccion_conos/tree/Test_Portatil

# CHECKS
- Weights in yolov5/weights/yolov5_models
- Active bash folder is Deteccion_conos
- Check requirements. Some might need to be installed with conda instead of pip
- ZED /usr/local/zed/get_python_api.py run to have pyzed library
- gcc compiler up to date for zed, conda install -c conda-forge gcc=12.1.0 # Otherwise zed library throws error: version `GLIBCXX_3.4.30' not found



# TODO
- KVASER
    https://www.kvaser.com/developer-blog/running-python-wrapper-linux/
    https://www.kvaser.com/developer-blog/kvaser-canlib-and-python-part-1-initial-setup/

"""

# CONSTANTS FOR SETTINGS (In the future move to globals/)
CAN_MODE = 0 # 0 -> CAN OFF, default values to test without CAN, 1 -> KVaser, 2 -> Arduino
CAMERA_MODE = 0 # 0 -> Webcam, 1 -> Read video file (VIDEO_FILE_NAME required), 2 -> ZED
VISUALIZE = 1

VIDEO_FILE_NAME = 'test_media/test_video.mp4' # Only used if CAMERA_MODE == 1
WEIGHTS_PATH = 'yolov5/weights/yolov5_models/240.pt'
#WEIGHTS_PATH = 'yolov5/weights/yolov5_models/TensorRT/240.engine' # TODO MAKE IT WORK with tensorrt weights?


# IMPORTS
import os
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt # For representation of time consumed

## Camera libraries
import cv2 # Webcam
import pyzed.sl as sl # ZED.

from connection_utils.car_comunication import ConnectionManager
from controller_agent.agent import AgentAccelerationYolo as AgentAcceleration
from cone_detection.yolo_detector import ConeDetector
from visualization_utils.visualizer_yolo_det import Visualizer
from visualization_utils.logger import Logger

# TODO add somewhere out of the thread
# Set the size with cv2.resize()

def read_image_cv2(cam,cam_queue):
    print(f'Starting camera thread...')
    recorded_times = np.array([0.]*3)
    
    while True:
        recorded_times[0] = time.time()
        result, image = cam.read() # TODO check if result == true?, CHECK cam.isOpened()?
        recorded_times[1] = time.time()
        cam_queue.put(image)
        recorded_times[2] = time.time()
        print(f'cam times: {[(recorded_times[i+1]-recorded_times[i]) for i in range(2)]}')

if __name__ == '__main__':
    # SETUP CAMERA
    if (CAMERA_MODE == 0):
        # WEBCAM

        cam = cv2.VideoCapture(0)
        
        # SETTINGS
        # cam.set(cv2.CAP_PROP_FPS, 60)
        # cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480) #1280 640 default
        # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #720 480 default
        # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        result, image = cam.read()
        while result == False:
            result, image = cam.read()
        np.array(image)
        
        # cam_queue  = multiprocessing.Queue(maxsize=1) #block=True, timeout=None
        # cam_worker  = multiprocessing.Process(target=read_image, args=(cam,cam_queue,), daemon=False)
        # cam_worker.start()
    elif (CAMERA_MODE == 1):
        # Read video file
        cam = cv2.VideoCapture(VIDEO_FILE_NAME)
        
    elif (CAMERA_MODE == 2):
        # Read ZED Camera - https://www.stereolabs.com/docs/video/camera-controls/
        
        cam = sl.Camera()
        
        # Camera settings
        cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 100) # We don't want blurry photos, we don't care about noise. The exposure time will still be adjusted automatically to compensate lighting conditions
        cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 8) # Maximum so it recognizes the color of the cones better. 0 to 8
        # cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 0) # Doesn't seem to make much difference. 0 to 8
        #cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1) # Fixed gain, so this is automatic. % of camera framerate (period)
        
        # Init parameters: https://www.stereolabs.com/docs/api/structsl_1_1InitParameters.html
        zed_params = sl.InitParameters()
        zed_params.camera_fps = 60
        zed_params.camera_resolution = sl.RESOLUTION.HD720 # HD1080 HD720 VGA (VGA=1344x376)
        #zed_params.sdk_gpu_id = -1 # Select which GPU to use. By default (-1) chooses most powerful NVidia
        
        
        status = cam.open(zed_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit(1)

        runtime = sl.RuntimeParameters()
        runtime.enable_depth = False # Deactivates de depth map calculation. We don't need it.
        
        # Create an RGBA sl.Mat object
        mat_img = sl.Mat()

    # INITIALIZE things
    ## Logger
    logger_path = os.path.join(os.getcwd(), "logs")
    init_message = "actuator_zed_testing.py"
    logger = Logger(logger_path, init_message)
    
    ## Cone detector
    detector = ConeDetector(checkpoint_path=WEIGHTS_PATH, logger=logger)
    
    ## Connections
    if (CAN_MODE == 1):
        connect_mng = ConnectionManager(logger=logger)
        print('CAN connection initialized')
    
    ## Agent
    agent = AgentAcceleration(logger=logger, target_speed=60.)

    ## Data visualization
    visualizer = Visualizer()


    # READ TIMES
    TIMES_TO_MEASURE = 5
    recorded_times = np.array([0.]*(TIMES_TO_MEASURE+2)) # Timetag at different points in code
    integrated_time_taken = np.array([0.]*TIMES_TO_MEASURE)
    average_time_taken = np.array([0.]*TIMES_TO_MEASURE)
    fps = -1.
    integrated_fps = 0.
    loop_counter = 0

    # Main loop #########################
    try:
        while True:
            recorded_times[0] = time.time()
            
            # Get CAN Data (To the car sensors or the simulator)
            if CAN_MODE == 1:
                # Get data from CAN
                in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(verbose=1)

            # GET IMAGE
            if (CAMERA_MODE == 0) or (CAMERA_MODE == 1):
                # WEBCAM or video file. Return image -> numpy array
                result, image = cam.read()
                while result == False:
                    result, image = cam.read()
                np.array(image)
                # image = cam_queue.get()
            elif (CAMERA_MODE == 2):
                # Read ZED camera
                if (cam.grab(runtime) == sl.ERROR_CODE.SUCCESS): # Grab gets the new frame
                    cam.retrieve_image(mat_img, sl.VIEW.LEFT) # Retrieve receives it and lets choose views and colormodes
                    image = mat_img.get_data() # Creates np.array()
            
            recorded_times[1] = time.time()
            # image = cv2.resize
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            recorded_times[2] = time.time()

            # Detect cones
            detections, cone_centers = detector.detect_cones(image, get_centers=True)
            recorded_times[3] = time.time()
            
            # Actions:
            # 1 -> steer
            # 2 -> throttle
            # 3 -> brake
            # 4 -> clutch
            # 5 -> upgear
            # 6 -> downgear

            # Get actions from agent
            [throttle, brake, steer, clutch, upgear, downgear, gear], data = agent.get_action(detections=detections,
                                                                                                speed=0,
                                                                                                gear=0,
                                                                                                rpm=0,
                                                                                                cone_centers=cone_centers,
                                                                                                image=image)
            recorded_times[4] = time.time()
            # resize actions
            # throttle *= 0.8
            # brake *= 0.8
            # steer *= 0.8
            # clutch *= 0.8

            # Send actions - CAN
            if (CAN_MODE == 1):
                connect_mng.send_actions(throttle=throttle,
                                        brake=brake,
                                        steer=steer,
                                        clutch=clutch,
                                        upgear=upgear,
                                        downgear=downgear)
            
            # VISUALIZE
            if (VISUALIZE == 1):
                cenital_map = [data[1], data[2], data[-1]]
                in_speed = 0
                in_gear = 0
                in_rpm = 0

                visualizer.visualize([image, detections, cone_centers,cenital_map, in_speed],
                                     [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm], fps,
                                     save_frames=False)
            
            recorded_times[5] = time.time()

            # END OF LOOP
            loop_counter += 1
            fps = 1/(recorded_times[5] - recorded_times[0])
            integrated_fps += fps
            integrated_time_taken += np.array([(recorded_times[i+1]-recorded_times[i]) for i in range(TIMES_TO_MEASURE)])

    finally:
        # When main loop stops, due to no image, error, Ctrl+C on terminal...

        # TIMES
        average_time_taken = integrated_time_taken/loop_counter # TODO remove 1
        fps = integrated_fps/loop_counter
        print(f'\n')
        print(f'FPS: {fps}')
        print(f'LOOPS: {loop_counter}')
        print(f'AVERAGE TIMES: {average_time_taken}')
        
        ## Plot the times
        fig = plt.figure(figsize=(12, 4))
        plt.bar(['cam.read()','cv2.cvtColor()','detect_cones()','agent.get_action()','visualize'],average_time_taken)
        plt.ylabel("Average time taken [s]")
        #plt.title("Title")
        #plt.show()
        plt.savefig("logs/times.png")

        # Close windows of visualizer
        #?
        
        
        # TODO CREATE CAR CLASS WITH THESE VARIABLES
        throttle = 0
        brake = 0
        steer = 0
        clutch = 0
        upgear = 0
        downgear = 0



