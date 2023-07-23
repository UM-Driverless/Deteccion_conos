# IMPORTS
import os, sys
import numpy as np
# import matplotlib.pyplot as plt # For representation of time consumed
# from canlib import canlib, Frame


## Our imports
from globals.globals import * # Global variables and constants, as if they were here


# FUNCTIONS
# def read_image_file(cam_queue):
#     """
#     Reads an image file
#     """
#     import cv2
    
#     print(f'Starting read_image_file thread...')
    
#     while True:
#         image = cv2.imread(IMAGE_FILE_NAME)
        
#         cam_queue.put(image)
        
# def read_image_video(cam_queue):
#     """
#     Reads a video file
#     """
#     import cv2
    
#     print(f'Starting read_image_video thread...')

#     cam = cv2.VideoCapture(VIDEO_FILE_NAME)
    
#     # SETTINGS
#     # cam.set(cv2.CAP_PROP_FPS, 60)
#     cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640) #1280 640 default
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
#     # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase
    
#     if (cam.isOpened() == False): 
#         print("Error opening video file")
    
#     while True:
#         # recorded_times_0 = time.time()
#         result, image = cam.read() # TODO also check CHECK cam.isOpened()?
#         while result == False:
#             result, image = cam.read()
        
#         # print(f'isOpened: {cam.isOpened()}')
#         # cv2.imshow('image',image)
#         # cv2.waitKey(10)
        
#         # recorded_times_1 = time.time()
        
#         cam_queue.put(image)
#         # print(f'Video read time: {recorded_times_1-recorded_times_0}')


# def read_image_webcam(cam_queue):
#     """
#     Reads the webcam
#     It usually takes about 35e-3 s to read an image, but in parallel it doesn't matter.
#     """
#     import cv2
    
#     print(f'Starting read_image_webcam thread...')

#     cam = cv2.VideoCapture(CAM_INDEX)
    
#     # SETTINGS
#     # cam.set(cv2.CAP_PROP_FPS, 60)
#     cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640) #1280 640 default
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
#     # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase
    
#     if (cam.isOpened() == False): 
#         print("Error opening webcam")
    
#     while True:
#         # recorded_times_0 = time.time()
        
#         # Read image from webcam
#         # TODO also check CHECK cam.isOpened()?
#         # It's 3 times faster if there are cones being detected. Nothing to do with visualize.
#         result, image = cam.read()
#         while result == False:
#             result, image = cam.read()
        
#         # recorded_times_1 = time.time()
        
#         # cv2.imshow('image',image)
#         # cv2.waitKey(1)
        
#         if FLIP_IMAGE:
#             image = cv2.flip(image, flipCode=1) # For testing purposes
        
#         cam_queue.put(image)
#         # print(f'Webcam read time: {recorded_times_1 - recorded_times_0}')


# def read_image_simulator(cam_queue, client):    
#     import fsds
#     import cv2
#     while True:
#         [img] = client.simGetImages([fsds.ImageRequest(camera_name = 'cam1', image_type = fsds.ImageType.Scene, pixels_as_float = False, compress = False)], vehicle_name = 'FSCar')
#         img_buffer = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
#         image = img_buffer.reshape(img.height, img.width, 3)
#         cam_queue.put(image)