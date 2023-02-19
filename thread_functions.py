# IMPORTS
# import numpy as np
# import matplotlib.pyplot as plt # For representation of time consumed
# from canlib import canlib, Frame

## Camera libraries
import cv2 # Webcam
import pyzed.sl as sl # ZED.

## Our imports
from globals.globals import * # Global variables and constants, as if they were here


# FUNCTIONS
def read_image_webcam():
    """Reads the webcam
    It usually takes about 35e-3 s to read an image, but in parallel it doesn't matter.
    """
    
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
        

def read_image_video(cam_queue):
    """Reads a video file
    """
    
    print(f'Starting read_image_video thread...')

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
    """Read the ZED camera - https://www.stereolabs.com/docs/video/camera-controls/
    """
    
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
    """
    Reads an image file
    """
    
    print(f'Starting read_image_file thread...')
    
    while True:
        image = cv2.imread(IMAGE_FILE_NAME)
        cam_queue.put(image)
    

def can_send_thread():
    print(f'Starting CAN receive thread...')
    global can_receive, can_queue
    
    while True:
        can_receive.receive_frame() # can_receive.frame updated
        # print(f'FRAME RECEIVED: {can_receive.frame}')
        # global car_state
        car_state_local = can_receive.new_state(car_state)
        # print(car_state_local)
        can_queue.put(car_state_local)
        
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
