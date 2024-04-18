import os, sys, cv2, time
import numpy as np
import multiprocessing
from abc import ABC, abstractmethod

class Camera(ABC):
    def __init__(self):
        self.process = None  # Initialize the process attribute
        self.cam_queue  = multiprocessing.Queue(maxsize=1) #block=True, timeout=None
        
        self.RESOLUTION = (640, 640) # (width, height) in pixels of the image given to net. Default yolo_v5 resolution is 640x640
        
        # FSDS_LIB_PATH = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python") # os.getcwd()
        self.SRC_DIR = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR = os.path.dirname(self.SRC_DIR)

    def start(self):
        """Start the process for capturing images."""
        
        if self.process is not None:
            return
        
        self.process = multiprocessing.Process(target=self.start_capture, args=(), daemon=True)
        self.process.start()

    def stop(self):
        """Stop the process safely, ensuring all resources are cleaned up."""
        
        cv2.destroyAllWindows()
        if self.process is not None:
            self.process.terminate()  # Send a signal to terminate the process
            self.process.join()  # Wait for the process to finish
            self.process = None

    @abstractmethod
    def start_capture(self):
        """Method to be implemented by subclasses for capturing images. This method will be run in a separate process.
        """
        pass
    
    def get_image(self):
        """Get the latest image from the camera queue."""
        image = self.cam_queue.get(timeout=4)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, self.RESOLUTION, interpolation=cv2.INTER_AREA)
        
        # CROP
            # height = len(image)
            # width = len(image[0])
            # image = np.array(image)[height-640:height, int(width/2)-320:int(width/2)+320]
        
        return image

    @staticmethod
    def create(mode):
        """Factory method to create a camera object based on the camera type.
        
        Example:
        camera = Camera.create(cam_queue)
        """
        if mode == 'image':
            return ImageFileCamera('test_media/cones_image.png')
        elif mode == 'video':
            return VideoFileCamera('test_media/video.mp4')
        elif mode == 'webcam':
            return WebcamCamera(0)
        elif mode == 'zed':
            return ZedOpenCVCamera()
        elif mode == 'sim':
            return SimulatorCamera()
        else:
            raise ValueError(f"Unknown camera type: {mode}")
    
    
class ImageFileCamera(Camera):
    def __init__(self, IMG_PATH = 'test_media/cones_image.png'):
        super().__init__()
        self.IMG_PATH = IMG_PATH

    def start_capture(self):
        """Run a thread to continuously read an image from a file, and put it into the queue"""
        print(f"Starting image file camera with file {self.IMG_PATH}")
        while True:
            image = cv2.imread(self.IMG_PATH)
            if image is not None:
                self.cam_queue.put(image)
            else:
                print(f"Failed to read image from {self.IMG_PATH}")
            time.sleep(0.1)  # Simulate a delay


class VideoFileCamera(Camera):
    def __init__(self, VID_PATH):
        super().__init__()
        self.VID_PATH = VID_PATH

    def start_capture(self):
        """Run a thread to continuously read frames from a video file, and put them into the queue"""
        print(f"Starting video file camera with file {self.VID_PATH}")
        cam = cv2.VideoCapture(self.VID_PATH)
        
        # Settings
        # cam.set(cv2.CAP_PROP_FPS, 60)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640) #1280 640 default
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
        # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase
        
        try: # TODO RETHINK THIS
            if (cam.isOpened() == False): 
                # print("ErrStarting read_image_zed (opencv) threador opening webcam")
                raise Exception("ERROR: Can't open video")
        except Exception as e:
            print(e)
            self.terminate()
            return
        
        while True:
            result, image = cam.read()
            while result == False:
                result, image = cam.read()
            self.cam_queue.put(image)

class WebcamCamera(Camera):
    def __init__(self, cam_index):
        super().__init__()
        self.cam_index = cam_index

    def start_capture(self):
        """Run a thread to continuously read frames from a webcam, and put them into the queue"""
        print(f"Starting WebcamCamera() with index {self.cam_index}...")
        cam = cv2.VideoCapture(self.cam_index)
        
        # Settings
        # cam.set(cv2.CAP_PROP_FPS, 60)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640) #1280 640 default
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
        # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase
        
        try: # TODO RETHINK THIS
            if (cam.isOpened() == False): 
                # print("ErrStarting read_image_zed (opencv) threador opening webcam")
                raise Exception("ERROR: Can't open video")
        except Exception as e:
            print(e)
            self.terminate()
            return
        
        while True:
            result, image = cam.read()
            while result == False:
                result, image = cam.read()
            self.cam_queue.put(image)


class ZedOpenCVCamera(Camera): # TODO TEST
    def __init__(self):
        super().__init__()
        
        import pyzed.sl as sl
        self.zed = sl.Camera()
        # self.zed_sensors = sl.SensorsData()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = 60
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    
    def start_capture(self):
        """Run a thread to continuously read frames from a ZED camera, and put them into the queue"""
        print(f"Starting ZedOpenCVCamera()...")
        
        cam = cv2.VideoCapture(CAM_INDEX)
        
        # SETTINGS
        # cam.set(cv2.CAP_PROP_FPS, 60)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640*2) #1280 640 default
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
        # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase
        
        try:
            if (cam.isOpened() == False): 
                # print("ErrStarting read_image_zed (opencv) threador opening webcam")
                raise Exception("ERROR: Can't open ZED as webcam")
        except Exception as e:
            print(e)
            self.terminate()
            return
        
        while True:
            result, image = cam.read()
            while result == False:
                result, image = cam.read()
            
            image = cv2.resize(image, (640*2,640), interpolation=cv2.INTER_AREA)
            image = np.array(image)[:,0:640,:]
            
            # image = cv2.flip(image, flipCode=1) # For testing purposes
            
            self.cam_queue.put(image)
            # print(f'Webcam read time: {recorded_times_1 - recorded_times_0}')
            
            ########## ZED SENSORS
            # zed.get_sensors_data(sensors,sl.TIME_REFERENCE.IMAGE)
            # quaternions = sensors.get_imu_data().get_pose().get_orientation().get()
            # state['orientation_y_rad'] = math.atan2(2*quaternions[1]*quaternions[3] - 2*quaternions[0]*quaternions[2], 1 - 2*quaternions[1]**2 - 2 * quaternions[2]**2)
    def stop(self):
        """Stop the process safely, ensuring all resources are cleaned up."""
        self.zed.close()
        cv2.destroyAllWindows()
        if self.process is not None:
            self.process.terminate()  # Send a signal to terminate the process
            self.process.join()  # Wait for the process to finish
            self.process = None
        


class SimulatorCamera(Camera):
    """
    Camera class for reading images from a simulator, and controlling the car from the resulting values. TODO MIXING BEHAVIOR, THINK. MAYBE ONE CLIENT WITH THIS CLASS, THE OTHER WITH COMM CLASS
    
    With https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator cloned in the home directory
    """
    def __init__(self):
        super().__init__()
        
        # TODO TRY THIRD CLIENT SO THE SIMULATOR AND MOUSE CAN WORK TOGETHER
    def start(self):
        """Start the process for capturing images."""
        
        if self.process is not None:
            return

        self.cam_queue = multiprocessing.Queue(maxsize=1)
        self.process = multiprocessing.Process(target=self.start_capture, args=(), daemon=True)
        self.process.start()
        
    def start_capture(self):
        """Run a thread to continuously read frames from a simulator, and put them into the queue.
        
        In theory this function is not picklable, because it is defined inside another function, and anything with self is not picklable. However, it seems to work as long as the client is defined inside the function. The client is not picklable, so it cannot be passed as an argument to the function. But the function itself can be in a thread while being a method of the class.
        """
        print(f"Starting SimulatorCamera()...")
    
        FSDS_LIB_PATH = os.path.join(os.path.dirname(self.ROOT_DIR), "Formula-Student-Driverless-Simulator", "python")
        sys.path.insert(0, FSDS_LIB_PATH)
        print(f'FSDS simulator path: {FSDS_LIB_PATH}')
        global fsds
        import fsds
        
        # connect to the simulator
        client = fsds.FSDSClient() # To get the image. WARNING: THIS IS NOT PICKLABLE, SO IT CANNOT BE PASSED TO A MULTIPROCESSING FUNCTION. HOWEVER, IT CAN BE DEFINED AND USED INSIDE THE WORKER FUNCTION ITSELF
        # Check network connection, exit if not connected
        client.confirmConnection()
        
        while True:
            [img] = client.simGetImages([fsds.ImageRequest(camera_name = 'cam1', image_type = fsds.ImageType.Scene, pixels_as_float = False, compress = False)], vehicle_name = 'FSCar')
            img_buffer = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
            image = img_buffer.reshape(img.height, img.width, 3)
            self.cam_queue.put(image)
    def stop(self):
        """Stop the process safely, ensuring all resources are cleaned up."""
        cv2.destroyAllWindows()
        if self.process is not None:
            self.process.terminate()
    
# TODO PUT IN COMM self.client.enableApiControl(False) # Allows mouse control, only API with this code
    
########## ZED SENSORS
# zed.get_sensors_data(sensors,sl.TIME_REFERENCE.IMAGE)
# quaternions = sensors.get_imu_data().get_pose().get_orientation().get()
# state['orientation_y_rad'] = math.atan2(2*quaternions[1]*quaternions[3] - 2*quaternions[0]*quaternions[2], 1 - 2*quaternions[1]**2 - 2 * quaternions[2]**2)

'''
import pyzed.sl as sl

print(f'Starting read_image_zed thread...', end='')
zed = sl.Camera()
print(f' (ZED SDK version: {zed.get_sdk_version()})')

# Init parameters: https://www.stereolabs.com/docs/api/structsl_1_1InitParameters.html
zed_params = sl.InitParameters()
sensors = sl.SensorsData() # TODO MOVE TO THREAD FOR HIGHER FREQUENCY? LINKED TO IMAGE SEEMS OK.
# zed_params.camera_fps = 100 # Not necessary. By default does max fps

# OPEN THE CAMERA
status = zed.open(zed_params)
while status != sl.ERROR_CODE.SUCCESS:
print(f'ZED ERROR: {status}')
status = zed.open(zed_params)
print('SUCESS, ZED opened')

# Camera settings
zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 100) # We don't want blurry photos, we don't care about noise. The exposure time will still be adjusted automatically to compensate lighting conditions
zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 8) # Maximum so it recognizes the color of the cones better. 0 to 8
# cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 0) # Doesn't seem to make much difference. 0 to 8
#cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1) # We have fixed gain, so this is automatic. % of camera framerate (period)

# RESOLUTION: HD1080 (3840x1080), HD720 (1280x720), VGA (VGA=1344x376)
# yolov5 uses 640x640. VGA is much faster, up to 100Hz
zed_params.camera_resolution = sl.RESOLUTION.HD720
zed_params.coordinate_units = sl.UNIT.METER
zed_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
print(f'ZED ORIENTATION VALUE: {sensors.get_imu_data().get_pose().get_orientation().get()}')
# state['orientation_y'] = sensors.get_imu_data().get_pose().get_orientation().get()
#zed_params.sdk_gpu_id = -1 # Select which GPU to use. By default (-1) chooses most powerful NVidia

runtime = sl.RuntimeParameters()
runtime.enable_depth = False # Deactivates depth map calculation. We don't need it.
zed_params.depth_mode = sl.DEPTH_MODE.NONE

# Create an RGBA sl.Mat object
mat_img = sl.Mat()

while True:
# Read ZED camera
if (zed.grab(runtime) == sl.ERROR_CODE.SUCCESS): # Grab gets the new frame
print('.')
# recorded_times_0 = time.time()

zed.retrieve_image(mat_img, sl.VIEW.LEFT) # Retrieve receives it and lets choose views and colormodes
image = mat_img.get_data() # Creates np.array()
self.cam_queue.put(image)

# recorded_times_1 = time.time()
# print(f'ZED read time: {recorded_times_1-recorded_times_0}')
zed.get_sensors_data(sensors,sl.TIME_REFERENCE.IMAGE)
quaternions = sensors.get_imu_data().get_pose().get_orientation().get()
state['orientation_y_rad'] = math.atan2(2*quaternions[1]*quaternions[3] - 2*quaternions[0]*quaternions[2], 1 - 2*quaternions[1]**2 - 2 * quaternions[2]**2)
'''



if __name__ == '__main__':
    IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_media', 'cones_image.png')
    VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_media', 'video.mp4')
    # camera = ImageFileCamera(IMAGE_PATH)
    # camera = VideoFileCamera(0)
    camera = SimulatorCamera()
    
    # os.startfile(r"C:\Users\ruben\Downloads\fsds-v2.2.0-windows\FSDS.exe")
    
    camera.start()
    
    try:
        # Display images continuously
        while True:
            cam_queue = camera.cam_queue
            if not cam_queue.empty():
                image = cam_queue.get()
                cv2.imshow('Camera Feed', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        camera.stop()
        cv2.destroyAllWindows()