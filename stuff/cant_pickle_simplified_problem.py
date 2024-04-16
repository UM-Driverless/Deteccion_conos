import os, sys, time, math, cv2, yaml, numpy as np
import multiprocessing

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
FSDS_LIB_PATH = os.path.join(os.path.dirname(ROOT_DIR), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, FSDS_LIB_PATH)
print(f'FSDS simulator path: {FSDS_LIB_PATH}')
global fsds
import fsds

def worker(client, cam_queue):
    import fsds
    while True:
        while True:
            [img] = client.simGetImages([fsds.ImageRequest(camera_name = 'cam1', image_type = fsds.ImageType.Scene, pixels_as_float = False, compress = False)], vehicle_name = 'FSCar')
            img_buffer = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
            image = img_buffer.reshape(img.height, img.width, 3)
            cam_queue.put(image)

if __name__ == '__main__':
    client = fsds.FSDSClient() # To get the image
    # Check network connection, exit if not connected
    client.confirmConnection()
    simulator_car_controls = fsds.CarControls()
    cam_queue = multiprocessing.Queue()
    
    process = multiprocessing.Process(target=worker, args=(client, cam_queue), daemon=True)
    process.start()
    
    # Show
    while True:
        if not cam_queue.empty():
            image = cam_queue.get()
            cv2.imshow('Camera Feed', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break