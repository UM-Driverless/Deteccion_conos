"""
Reads an image file
"""
import cv2
import numpy as np

print(f'Starting read_image_file thread...')
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640) #1280 640 default
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default

while True:
    # image = cv2.imread('test_media/image1.png')
    result, image = cam.read()
    # print(result)
    
    image = cv2.resize(image, (640*2,640), interpolation=cv2.INTER_AREA)
    image = np.array(image)[:,0:int(len(image[0])/2),:]
    
    cv2.imshow('im',image)
    cv2.waitKey(0)
    
