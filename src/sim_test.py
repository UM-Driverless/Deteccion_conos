import os, sys, time, math, cv2, yaml, numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
FSDS_LIB_PATH = os.path.join(os.path.dirname(ROOT_DIR), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, FSDS_LIB_PATH)
print(f'FSDS simulator path: {FSDS_LIB_PATH}')
global fsds
import fsds

# connect to the simulator
sim_client1 = fsds.FSDSClient() # To get the image
# TODO TRY THIRD CLIENT SO THE SIMULATOR AND MOUSE CAN WORK TOGETHER

# Check network connection, exit if not connected
sim_client1.confirmConnection()

simulator_car_controls = fsds.CarControls()

import fsds
while True:
    while True:
        [img] = sim_client1.simGetImages([fsds.ImageRequest(camera_name = 'cam1', image_type = fsds.ImageType.Scene, pixels_as_float = False, compress = False)], vehicle_name = 'FSCar')
        img_buffer = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
        image = img_buffer.reshape(img.height, img.width, 3)
        # show
        cv2.imshow('Camera Feed', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break