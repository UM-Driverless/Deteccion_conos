import cv2 # Webcam

print(f'Starting read_image_file thread...')
    
# cam = cv2.VideoCapture('test_media/cones_image.png')

# SETTINGS
# # cam.set(cv2.CAP_PROP_FPS, 60)
# cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640) #1280 640 default
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
# # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase

# if (cam.isOpened() == False): 
#     print("Error opening image file")

while True:
    # recorded_times_0 = time.time()
    image = cv2.imread('test_media/cones_image.png')
    # while result == False:
    #     result, image = cam.read()
    
    # print(f'isOpened: {cam.isOpened()}')
    cv2.imshow('image',image)
    cv2.waitKey(10)
    print('.\n')
    # recorded_times_1 = time.time()
    
    # cam_queue.put(image)
    # print(f'Video read time: {recorded_times_1-recorded_times_0}')