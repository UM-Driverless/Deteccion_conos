
# MAIN script to execute all the others. Shall contain the main classes, top level functions etc.

"""
# HOW TO USE
- Configuration variables in globals.py


# REFERENCES
https://github.com/UM-Driverless/Deteccion_conos/tree/Test_Portatil
vulture . --min-confidence 100

# CHECKS
- Weights in yolov5/weights/yolov5_models
- Active bash folder is Deteccion_conos
- Check requirements. Some might need to be installed with conda instead of pip
- ZED /usr/local/zed/get_python_api.py run to have pyzed library
- gcc compiler up to date for zed, conda install -c conda-forge gcc=12.1.0 # Otherwise zed library throws error: version `GLIBCXX_3.4.30' not found

# TODO
- Kvaser CAN!!!!
- Print number of cones detected (per color or total)
- Xavier why network takes 3s to execute.
- Better color recognition
- Make net faster. Remove cone types that we don't use?
- Check NVPMODEL with high power during xavier installation
- KVASER
    https://www.kvaser.com/developer-blog/running-python-wrapper-linux/
    https://www.kvaser.com/developer-blog/kvaser-canlib-and-python-part-1-initial-setup/

- Can't install anaconda on xavier, it says it's not compatible due to arch64 architecture.

- Wanted to make visualize work in a thread and for any resolution, but now it works for any resolution, don't know why, and it's always about 3ms so it's not worth it for now.

# STUFF
#if __name__ == '__main__': # removed because this file should never be imported as a module.
To stop: Ctrl+C in the terminal

"""

# IMPORTS
import os
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt # For representation of time consumed
import sys
print(f'Python version: {sys.version}')

## Camera libraries
import cv2 # Webcam
import pyzed.sl as sl # ZED.

## Our imports
from globals import * # Global variables and constants, as if they were here
from connection_utils.car_comunication import ConnectionManager
from controller_agent.agent import AgentAccelerationYolo as AgentAcceleration
from cone_detection.yolo_detector import ConeDetector
from visualization_utils.visualizer_yolo_det import Visualizer
from visualization_utils.logger import Logger

cam_queue  = multiprocessing.Queue(maxsize=1) #block=True, timeout=None. Global variable

def read_image_webcam():
    '''Reads the webcam
    It usually takes about 35e-3 s to read an image, but in parallel it doesn't matter.
    '''
    
    print(f'Starting read_image_webcam thread...')
    
    global cam_queue # To access the global cam_queue instead of a local copy

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
        
        cam_queue.put(image)
        # print(f'Webcam read time: {recorded_times_1 - recorded_times_0}')
        

def read_image_video():
    '''Reads a video file
    '''
    
    print(f'Starting read_image_video thread...')
    
    global cam_queue # Required only to modify cam_queue. We don't. Just in case

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
        # recorded_times_0 = time.time()
        result, image = cam.read() # TODO also check CHECK cam.isOpened()?
        while result == False:
            result, image = cam.read()
        
        # print(f'isOpened: {cam.isOpened()}')
        # cv2.imshow('image',image)
        # cv2.waitKey(10)
        
        # recorded_times_1 = time.time()
        
        cam_queue.put(image)
        # print(f'Video read time: {recorded_times_1-recorded_times_0}')


def read_image_zed():
    '''Read the ZED camera - https://www.stereolabs.com/docs/video/camera-controls/
    '''
    
    print(f'Starting read_image_zed thread...')
    
    global runtime, cam_queue # Required only to modify. We don't. Just in case
    
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
    zed_params.camera_resolution = sl.RESOLUTION.VGA
    
    #zed_params.sdk_gpu_id = -1 # Select which GPU to use. By default (-1) chooses most powerful NVidia
    
    status = cam.open(zed_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f'ZED ERROR: {repr(status)}')
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

def read_image_file():
    '''
    Reads an image file
    '''
    
    print(f'Starting read_image_file thread...')
    
    while True:
        image = cv2.imread(IMAGE_FILE_NAME)
        cam_queue.put(image)
    

def agent_thread():
    print(f'Starting agent thread...')
    global agent_target, detections, cone_centers, image
    
    while True:
        # This agent_target variable must be local, and sent to the main loop through a queue that manages the concurrency.
        [agent_target_local, data] = agent.get_action(agent_target,
                                                      car_state,
                                                      detections=detections,
                                                      cone_centers=cone_centers,
                                                      image=image)
        
        # Output values to queue as an array
        agent_queue.put([agent_target_local, data])

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

'''
# Visualize thread directly here
def visualize_thread():
    print(f'Starting visualize thread...')
    while True:        
        # image, detections, cone_centers, cenital_map, speed = [image, detections, cone_centers,cenital_map, in_speed]
        bbox, labels = detections
        # cenital_map, estimated_center, wrap_img = cenital_map
        # throttle, brake, steer, clutch, upgear, downgear, gear, rpm = [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm]
        
        image = cam_queue.get(block=False, timeout=5) # Read an image but don't remove it. Only the main loop takes it
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Color values of each cone type, in bgr
        colors = {
            'blue_cone': (255, 0, 0),
            'yellow_cone': (0, 255, 255),
            'orange_cone': (40, 50, 200), #(40, 50, 200)
            'large_orange_cone': (40, 100, 255), #(40, 100, 255)
            'unknown_cone': (0,0,0)
        }
        
        # Print boxes around each detected cone
        image = Visualizer.print_bboxes(image, bbox, labels, colors)

        # Print cenital map
        # image = self._print_cenital_map(cenital_map, colors, estimated_center, image) # TODO MAKE IT WORK

        # Print the output values of the agent, trying to control the car
        # image = Visualizer.print_data(0, 0, fps, 0, image, 0, 0, 0, 0, len(labels))

        

        # dim = (np.array(image.shape) * 0.1).astype('int')
        # image[400:400 + dim[1], 10:10 + dim[1]] = cv2.resize(wrap_img, (dim[1], dim[1]))

        #TODO make faster or in parallel #takestime
        cv2.imshow("Detections", image)
        cv2.waitKey(100)
'''


# SETUP CAMERA
if (CAMERA_MODE == 0):   cam_worker = multiprocessing.Process(target=read_image_webcam, args=(), daemon=False)
elif (CAMERA_MODE == 1): cam_worker = multiprocessing.Process(target=read_image_zed,    args=(), daemon=False)
elif (CAMERA_MODE == 2): cam_worker = multiprocessing.Process(target=read_image_file,   args=(), daemon=False)
elif (CAMERA_MODE == 3): cam_worker = multiprocessing.Process(target=read_image_video,  args=(), daemon=False)

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
recorded_times = np.array([0.]*(TIMES_TO_MEASURE+2)) # Timetags at different points in code
integrated_time_taken = np.array([0.]*TIMES_TO_MEASURE)
average_time_taken = np.array([0.]*TIMES_TO_MEASURE)
fps = -1.
integrated_fps = 0.
loop_counter = 0

## Data visualization
if (VISUALIZE == 1):
    visualizer = Visualizer()
    # visualize_worker = multiprocessing.Process(target=visualize_thread, args=(), daemon=False)


# Main loop ------------------------
try:
    print(f'Starting main loop...')
    while True:
        recorded_times[0] = time.time()
        
        # Get CAN Data (To the car sensors or the simulator)
        if CAN_MODE == 1:
            # Get data from CAN
            in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(verbose=1)

        # GET IMAGE (Either from webcam, video, or ZED camera)
        image = cam_queue.get(timeout=5)
        
        # Resize to IMAGE_RESOLUTION no matter how we got the image
        image = cv2.resize(image, IMAGE_RESOLUTION, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.flip(image, flipCode=1) # For testing purposes
        # image = np.array(image)
        
        recorded_times[1] = time.time()

        # Detect cones
        detections, cone_centers = detector.detect_cones(image, get_centers=True)
        
        recorded_times[2] = time.time()
        
        # Get actions from agent
        if (loop_counter == 0):
            agent_worker.start()
            # visualize_worker.start() #Visualizer
        
        [agent_target, data] = agent_queue.get()
        
        # Test with a plot
        # if (len(cone_centers[0]) > 0):
        #     plt.figure('cone_centers')
        #     plt.clf()
        #     plt.scatter(data[0][0][:,0], data[0][0][:,1])
        #     plt.scatter(data[0][1][:,0], data[0][1][:,1])
        #     plt.show()
        #     plt.savefig("logs/cone_centers.png")
        
        
        recorded_times[3] = time.time()

        # Send actions - CAN
        if (CAN_MODE == 1):
            connect_mng.send_actions(throttle = agent_target['throttle'],
                                    brake = agent_target['brake'],
                                    steer = agent_target['steer'],
                                    clutch = agent_target['clutch'])
        
        # VISUALIZE
        # TODO add parameters to class
        if (VISUALIZE == 1):
            cenital_map = [data[1], data[2], data[-1]]
            in_speed = 0
            in_rpm = 0

            visualizer.visualize(agent_target,
                                 car_state,
                                 image,
                                 detections,
                                 fps,
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

    # Close processes and windows
    cam_worker.terminate()
    agent_worker.terminate()
    cv2.destroyAllWindows()    
    
    agent_target = {
        "throttle": 0.,
        "brake": 0.,
        "steer": 0.,
        "clutch": 0.,
    }
