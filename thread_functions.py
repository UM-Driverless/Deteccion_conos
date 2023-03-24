# IMPORTS
import os, sys
import numpy as np
# import matplotlib.pyplot as plt # For representation of time consumed
# from canlib import canlib, Frame


## Our imports
from globals.globals import * # Global variables and constants, as if they were here


# FUNCTIONS
def read_image_file(cam_queue):
    """
    Reads an image file
    """
    import cv2
    
    print(f'Starting read_image_file thread...')
    
    while True:
        image = cv2.imread(IMAGE_FILE_NAME)
        cam_queue.put(image)
        
def read_image_video(cam_queue):
    """
    Reads a video file
    """
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


def read_image_webcam(cam_queue):
    """
    Reads the webcam
    It usually takes about 35e-3 s to read an image, but in parallel it doesn't matter.
    """
    import cv2
    
    print(f'Starting read_image_webcam thread...')

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


def read_image_simulator(cam_queue):
    fsds_lib_path = os.path.join(os.getcwd(),"Formula-Student-Driverless-Simulator","python")
    sys.path.insert(0, fsds_lib_path)
    # print(f'SIMULATOR CAMERA FSDS PATH: {fsds_lib_path}')
    import fsds # TODO why not recognized when debugging
    
    # connect to the simulator 
    client = fsds.FSDSClient()

    # Check network connection, exit if not connected
    client.confirmConnection()
    
    while True:
        [img] = client.simGetImages([fsds.ImageRequest(camera_name = 'examplecam', image_type = fsds.ImageType.Scene, pixels_as_float = False, compress = False)], vehicle_name = 'FSCar')
        img_buffer = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
        image = img_buffer.reshape(img.height, img.width, 3)
        cam_queue.put(image)
        i = i+1


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
