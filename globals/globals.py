# CONSTANTS FOR SETTINGS
CAN_MODE = 0 # 0 -> CAN OFF, default values to test without CAN, 1 -> KVaser, 2 -> Arduino
CAMERA_MODE = 4 # 0 -> Image file, 1 -> Read video file (VIDEO_FILE_NAME required), 2 -> Webcam, 3 -> ZED, 4 -> SIMULATOR, 5 -> Simulator (manual control)
# Choose webcam
CAM_INDEX = 0 # `ls /dev/video*` to check number. With ZED: one opens both, the other doesn't work.

VISUALIZE = 1
VISUALIZER_CENITAL_MAP_SIZE_PERC = 0.5

# MISSION SELECTION CONSTANTS
MISSION_SELECTED = 0 # 0 -> Generic: Runs continuously, 1 -> Acceleration, 2 -> Skidpad, 3 -> Autocross, 4 -> Trackdrive, 5 -> EBS Test, ... (Using the example of the begginers guide)

# CAM CONSTANTS
CAMERA_VERTICAL_FOV_DEG = 70 # 120 the horizontal FOV
CAM_HEIGHT = 0.75 # m
CAM_HORIZON_POS = 0.5 # per 1 of image from top
# Simulator camera (cam1) pos: (-.3,-.16,.8)


# For video file
VIDEO_FILE_NAME = 'test_media/videosim.mp4' # Only used if CAMERA_MODE == 1
IMAGE_FILE_NAME = 'test_media/cones_image.png'
# IMAGE_FILE_NAME = 'test_media/image3.webp'
WEIGHTS_PATH = 'yolov5/weights/yolov5_models/best_adri.pt'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/280.engine'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/280_adri.pt'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/240.pt'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/TensorRT/240.engine' # TODO MAKE IT WORK with tensorrt weights?
IMAGE_RESOLUTION = (640, 640) # (width, height) in pixels of the image given to net. Default yolo_v5 resolution is 640x640

CONFIDENCE_THRESHOLD = 0.5 # 0.5
FLIP_IMAGE = 0

# CAN
CAN_SEND_MSG_TIMEOUT = 0.005
CAN_ACTION_DIMENSION = 100.
CAN_STEER_DIMENSION = 122880.

# ---

# Actual state values of the car, from sensors
car_state = {
    "speed": 0.,
    "gear": 0.,
    "rpm": 0.,
    "speed_senfl": 0.,
    "speed_senfl": 0.
}

# Target values obtained from agent.get_action()
agent_target = {
    "acc": 0., # Acceleration. From -1.0 to 1.0. TODO Then translates into throttle and brake
    "steer": 0., # -1.0 to 1.0
    "throttle": 0., # 0.0 to 1.0
    "brake": 0., # 0.0 to 1.0
    # "clutch": 0., # ? 0.8 I've seen
    # "upgear": 0., # 
    # "downgear": 0.,
    # "gear": 0. # Should be an integer
}

# Neural net detections [{bounding boxes},{labels}] = [ [[[x1,y1],[x2,y2]], ...], [[{class name}, {confidence}], ...] ]
detections = []
