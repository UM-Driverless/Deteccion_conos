
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


# STUFF
#if __name__ == '__main__': # removed because this file shouldn never be imported as a module.

"""

# CONSTANTS FOR SETTINGS (In the future move to globals/)
CAN_MODE = 0 # 0 -> CAN OFF, default values to test without CAN, 1 -> KVaser, 2 -> Arduino
CAMERA_MODE = 2 # 0 -> Webcam, 1 -> Read video file (VIDEO_FILE_NAME required), 2 -> ZED
VISUALIZE = 1

# For webcam
CAM_INDEX = 0
# For video file
VIDEO_FILE_NAME = 'test_media/video_short.mp4' # Only used if CAMERA_MODE == 1
WEIGHTS_PATH = 'yolov5/weights/yolov5_models/240.pt'
#WEIGHTS_PATH = 'yolov5/weights/yolov5_models/TensorRT/240.engine' # TODO MAKE IT WORK with tensorrt weights?
IMAGE_RESOLUTION = (640, 640) # (width, height) in pixels. Default yolo_v5 resolution is 640x640


# IMPORTS
import os
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt # For representation of time consumed

## Camera libraries
import cv2 # Webcam
import pyzed.sl as sl # ZED.

## Our imports
from connection_utils.car_comunication import ConnectionManager
from controller_agent.agent import AgentAccelerationYolo as AgentAcceleration
from cone_detection.yolo_detector import ConeDetector
from visualization_utils.visualizer_yolo_det import Visualizer
from visualization_utils.logger import Logger

# TODO add somewhere out of the thread
# Set the size with cv2.resize()

cam_queue  = multiprocessing.Queue(maxsize=1) #block=True, timeout=None. Global variable

def read_image_webcam():
    '''Reads the webcam
    It usually takes about 35e-3 s to read an image, but in parallel
    '''
    
    print(f'Starting read_image_webcam thread...')
    
    # global cam_queue # TODO necessary to declare here?

    cam = cv2.VideoCapture(CAM_INDEX)
    
    # SETTINGS
    # cam.set(cv2.CAP_PROP_FPS, 60)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640) #1280 640 default
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
    # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase
    
    if (cam.isOpened() == False): 
        print("Error opening webcam")
    
    while True:
        recorded_times_0 = time.time()
        result, image = cam.read() # TODO also check CHECK cam.isOpened()?
        while result == False:
            result, image = cam.read()
        
        recorded_times_1 = time.time()
        
        cam_queue.put(image)
        print(f'Webcam read time: {recorded_times_1-recorded_times_0}')


def read_image_video():
    '''Reads a video file
    '''
    
    print(f'Starting read_image_video thread...')
    
    global cam_queue # TODO necessary to declare here?

    cam = cv2.VideoCapture(VIDEO_FILE_NAME)
    
    # SETTINGS
    # cam.set(cv2.CAP_PROP_FPS, 60)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640) #1280 640 default
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
    # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase
    
    if (cam.isOpened() == False): 
        print("Error opening video file")
    
    while True:
        recorded_times_0 = time.time()
        result, image = cam.read() # TODO also check CHECK cam.isOpened()?
        while result == False:
            result, image = cam.read()
        
        # print(f'isOpened: {cam.isOpened()}')
        # cv2.imshow('image',image)
        # cv2.waitKey(10)
        
        recorded_times_1 = time.time()
        
        cam_queue.put(image)
        print(f'Video read time: {recorded_times_1-recorded_times_0}')


def read_image_zed():
    '''Read the ZED camera - https://www.stereolabs.com/docs/video/camera-controls/
    '''
    
    print(f'Starting read_image_zed thread...')
    
    global runtime, cam_queue # Only read
    
    cam = sl.Camera()
    
    # Camera settings
    cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 100) # We don't want blurry photos, we don't care about noise. The exposure time will still be adjusted automatically to compensate lighting conditions
    cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 8) # Maximum so it recognizes the color of the cones better. 0 to 8
    # cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 0) # Doesn't seem to make much difference. 0 to 8
    #cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1) # Fixed gain, so this is automatic. % of camera framerate (period)
    
    # Init parameters: https://www.stereolabs.com/docs/api/structsl_1_1InitParameters.html
    zed_params = sl.InitParameters()
    # zed_params.camera_fps = 100 # Not necessary. By default does max fps
    
    # RESOLUTION: HD1080 (3840x1080), HD720 (1280x720), VGA (VGA=1344x376)
    # yolov5 uses 640x640. VGA is much faster, up to 100Hz
    zed_params.camera_resolution = sl.RESOLUTION.HD720
    
    #zed_params.sdk_gpu_id = -1 # Select which GPU to use. By default (-1) chooses most powerful NVidia
    
    status = cam.open(zed_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    runtime = sl.RuntimeParameters()
    runtime.enable_depth = False # Deactivates de depth map calculation. We don't need it.
    
    # Create an RGBA sl.Mat object
    mat_img = sl.Mat()
    
    while True:
        # Read ZED camera
        if (cam.grab(runtime) == sl.ERROR_CODE.SUCCESS): # Grab gets the new frame
            # recorded_times_0 = time.time()
            
            cam.retrieve_image(mat_img, sl.VIEW.LEFT) # Retrieve receives it and lets choose views and colormodes
            image = mat_img.get_data() # Creates np.array()
            cam_queue.put(image)
            
            # recorded_times_1 = time.time()
            # print(f'ZED read time: {recorded_times_1-recorded_times_0}')
        

def agent_thread():
    print(f'Starting agent thread...')
    global detections, cone_centers, image # Only read
    
    while True:
        [[throttle, brake, steer, clutch, upgear, downgear, gear], data] = agent.get_action(detections=detections,
                                                                                        speed=0,
                                                                                        gear=0,
                                                                                        rpm=0,
                                                                                        cone_centers=cone_centers,
                                                                                        image=image)
        
        # Output values to queue as an array
        agent_queue.put([[throttle, brake, steer, clutch, upgear, downgear, gear], data])

''' Visualize thread doesn't work. It's not required for the car to work so ignore it.
# def visualize_thread():
#     print(f'Starting visualize thread...')
#     global visualizer
#     global image, detections, cone_centers, cenital_map, in_speed
#     global throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm, fps
    
#     print(visualizer)
#     visualizer.visualize([image, detections, cone_centers, cenital_map, in_speed],
#                         [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm], fps,
#                         save_frames=False)
'''

# SETUP CAMERA
if (CAMERA_MODE == 0):   cam_worker = multiprocessing.Process(target=read_image_webcam, args=(), daemon=False)
elif (CAMERA_MODE == 1): cam_worker = multiprocessing.Process(target=read_image_video,  args=(), daemon=False)
elif (CAMERA_MODE == 2): cam_worker = multiprocessing.Process(target=read_image_zed,    args=(), daemon=False)

cam_worker.start()

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
agent_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None
agent_worker = multiprocessing.Process(target=agent_thread, args=(), daemon=False)

# READ TIMES
TIMES_TO_MEASURE = 4
recorded_times = np.array([0.]*(TIMES_TO_MEASURE+2)) # Timetag at different points in code
integrated_time_taken = np.array([0.]*TIMES_TO_MEASURE)
average_time_taken = np.array([0.]*TIMES_TO_MEASURE)
fps = -1.
integrated_fps = 0.
loop_counter = 0

## Data visualization
if (VISUALIZE == 1):
    visualizer = Visualizer()


# Main loop ------------------------
try:
    while True:
        recorded_times[0] = time.time()
        
        # Get CAN Data (To the car sensors or the simulator)
        if CAN_MODE == 1:
            # Get data from CAN
            in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(verbose=1)

        # GET IMAGE (Either from webcam, video, or ZED camera)
        image = cam_queue.get(timeout=5)
        
        # Resize to IMAGE_RESOLUTION no matter how we got the image
        # image = cv2.resize(image, IMAGE_RESOLUTION, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = np.array(image)
        
        recorded_times[1] = time.time()

        # Detect cones
        detections, cone_centers = detector.detect_cones(image, get_centers=True)
        recorded_times[2] = time.time()
        
        # Get actions from agent
        ## Actions:
            # 1 -> steer
            # 2 -> throttle
            # 3 -> brake
            # 4 -> clutch
            # 5 -> upgear
            # 6 -> downgear

        # [throttle, brake, steer, clutch, upgear, downgear, gear], data = agent.get_action(detections=detections,
        #                                                                                     speed=0,
        #                                                                                     gear=0,
        #                                                                                     rpm=0,
        #                                                                                     cone_centers=cone_centers,
        #                                                                                     image=image)
        
        if (loop_counter == 0):
            agent_worker.start()
        
        [[throttle, brake, steer, clutch, upgear, downgear, gear], data] = agent_queue.get()
        
        recorded_times[3] = time.time()

        # Send actions - CAN
        if (CAN_MODE == 1):
            connect_mng.send_actions(throttle=throttle,
                                    brake=brake,
                                    steer=steer,
                                    clutch=clutch,
                                    upgear=upgear,
                                    downgear=downgear)
        
        # VISUALIZE
        # TODO add parameters to class
        if (VISUALIZE == 1):
            cenital_map = [data[1], data[2], data[-1]]
            in_speed = 0
            in_gear = 0
            in_rpm = 0

            visualizer.visualize([image, detections, cone_centers,cenital_map, in_speed],
                                    [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm], fps,
                                    save_frames=False)
        
        recorded_times[4] = time.time()

        # END OF LOOP
        loop_counter += 1
        fps = 1/(recorded_times[TIMES_TO_MEASURE] - recorded_times[0])
        integrated_fps += fps
        integrated_time_taken += np.array([(recorded_times[i+1]-recorded_times[i]) for i in range(TIMES_TO_MEASURE)])

finally:
    # When main loop stops, due to no image, error, Ctrl+C on terminal, this calculates performance metrics and closes everything.

    # TIMES
    # cam.release()
    cv2.destroyAllWindows()
    if loop_counter != 0:
        average_time_taken = integrated_time_taken/loop_counter
        fps = integrated_fps/loop_counter
    else:
        average_time_taken = -1
        fps = -1
    print(f'\n\n\n------------RESULTS------------\n',end='')
    print(f'FPS: {fps}')
    print(f'LOOPS: {loop_counter}')
    print(f'AVERAGE TIMES: {average_time_taken}')
    print(f'-------------------------------\n',end='')
    
    ## Plot the times
    fig = plt.figure(figsize=(12, 4))
    plt.bar(['cam.read()','detect_cones()','agent.get_action()','visualize'],average_time_taken)
    plt.ylabel("Average time taken [s]")
    plt.figtext(.8,.8,f'{fps:.2f}Hz')
    plt.title("Execution time per section of main loop")
    plt.savefig("logs/times.png")

    # Close windows of visualizer?
    
    
    # TODO CREATE CAR CLASS WITH THESE VARIABLES
    throttle = 0
    brake = 0
    steer = 0
    clutch = 0
    upgear = 0
    downgear = 0



