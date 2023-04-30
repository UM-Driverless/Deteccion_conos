# CONSTANTS FOR SETTINGS
CAN_MODE = 0 # 0 -> CAN OFF, default values to test without CAN, 1 -> KVaser, 2 -> Arduino
CAMERA_MODE = 3 # 0 -> Image file, 1 -> Read video file (VIDEO_FILE_NAME required), 2 -> Webcam, 3 -> ZED, 4 -> SIMULATOR
SIMULATOR_CONTROL = 1

VISUALIZE = 1
VISUALIZER_CENITAL_MAP_SIZE_PERC = 0.5

# CAM CONSTANTS
CAMERA_FOV_DEG = 120
CAM_HEIGHT = 0.95 # m
CAM_HORIZON_POS = 0.5 # % of image from top

# Choose webcam
CAM_INDEX = 0 # `ls /dev/video*` to check number. With ZED: one opens both, the other doesn't work.

# For video file
VIDEO_FILE_NAME = 'test_media/video.mp4' # Only used if CAMERA_MODE == 1
IMAGE_FILE_NAME = 'test_media/image1.png'
# IMAGE_FILE_NAME = 'test_media/image3.webp'
WEIGHTS_PATH = 'yolov5/weights/yolov5_models/best_adri.pt'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/280.engine'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/280_adri.pt'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/240.pt'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/TensorRT/240.engine' # TODO MAKE IT WORK with tensorrt weights?
IMAGE_RESOLUTION = (640, 640) # (width, height) in pixels of the image given to net. Default yolo_v5 resolution is 640x640

CONFIDENCE_THRESHOLD = 0.5 # 0.5

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
    "clutch": 0., # ? 0.8 I've seen
    # "upgear": 0., # 
    # "downgear": 0.,
    # "gear": 0. # Should be an integer
}

# Neural net detections [{bounding boxes},{labels}] = [ [[[x1,y1],[x2,y2]], ...], [[{class name}, {confidence}], ...] ]
detections = []
