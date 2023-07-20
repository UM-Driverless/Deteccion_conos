fps = -1.0

# CONSTANTS FOR SETTINGS
CAN_MODE = 0 # 0 -> CAN OFF, default values to test without CAN, 1 -> KVaser, 2 -> Arduino
CAMERA_MODE = 4 # 0 -> Image file, 1 -> Read video file (VIDEO_FILE_NAME required), 2 -> Webcam, 3 -> ZED, 4 -> SIMULATOR, 5 -> Simulator (manual control)
# Choose webcam
CAM_INDEX = 0 # `ls /dev/video*` to check number. With ZED: one opens both, the other doesn't work.

VISUALIZE = 1
VISUALIZER_CENITAL_MAP_SIZE_PERC = 0.5

# MISSION SELECTION CONSTANTS
MISSION_SELECTED = 0 # 0 -> Generic: Runs continuously, 1 -> Acceleration, 2 -> Skidpad, 3 -> Autocross, 4 -> Trackdrive, 5 -> EBS Test, ... (Using the example of the begginers guide)

# Test media addresses
VIDEO_FILE_NAME = 'test_media/videosim.mp4' # Only used if CAMERA_MODE == 1
IMAGE_FILE_NAME = 'test_media/cones_image.png'
# IMAGE_FILE_NAME = 'test_media/image3.webp'

# CAM CONSTANTS
CAMERA_VERTICAL_FOV_DEG = 70 # 120 the horizontal FOV
CAM_HEIGHT = 0.75 # m
CAM_HORIZON_POS = 0.5 # per 1 of image from top
# Simulator camera (cam1) pos: (-.3,-.16,.8)


WEIGHTS_PATH = 'yolov5/weights/yolov5_models/best_adri.pt'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/280.engine'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/280_adri.pt'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/240.pt'
# WEIGHTS_PATH = 'yolov5/weights/yolov5_models/TensorRT/240.engine' # TODO MAKE IT WORK with tensorrt weights?
IMAGE_RESOLUTION = (640, 640) # (width, height) in pixels of the image given to net. Default yolo_v5 resolution is 640x640

CONFIDENCE_THRESHOLD = 0.5 # 0.5
FLIP_IMAGE = 0

# CAN
CAN_SEND_TIMEOUT = 0.005
CAN_ACTION_DIMENSION = 100.
CAN_STEER_DIMENSION = 122880.

# ---

# Actual state values of the car, from sensors
car_state = {
    'speed': 0., # m/s, can be calculated from speed_senfl mixed with other sensors.
    'rpm': 0.,
    'speed_senfl': 0.,
}

# Target values obtained from agent.get_action()
agent_target = {
    'acc': 0., # Acceleration. From -1.0 to 1.0. TODO Then translates into throttle and brake
    'steer': 0., # -1.0 to 1.0
    'throttle': 0., # 0.0 to 1.0
    'brake': 0., # 0.0 to 1.0
}

# Neural net detections [{bounding boxes},{labels}] = [ [[[x1,y1],[x2,y2]], ...], [[{class name}, {confidence}], ...] ]
detections = []


# CAN CONSTANTS (ID, Message templates to add data etc.)
CAN = {
    'SENFL': { # Front Left wheel
        'IMU': int('300', 16), # 8 bytes - [Ax Ay Az Gx Gy Gz Mx My] TODO WRITE ID, its an id
        'SIG': int('301', 16), # 8 bytes - [Analog1 Analog2 Analog3 Digital Rel1 Rel2 Rel3]
        'STATE': int('302', 16), # 8 bytes - [Error id Null Null Null Null Null Null Null]
    },
    'SENFR': { # Front Right wheel
        'IMU': int('303', 16),
        'SIG': int('304', 16),
        'STATE': int('305', 16),
    },
    'SENRL': { # Rear Left wheel (UNUSED)
        'IMU': int('306', 16),
        'SIG': int('307', 16),
        'STATE': int('308', 16),
    },
    'SENRR': { # Rear Right wheel (UNUSED)
        'IMU': int('309', 16),
        'SIG': int('310', 16),
        'STATE': int('311', 16),
    },
    'TRAJ': {
        'ACT': int('320', 16),
        'GPS': int('321', 16),
        'IMU': int('322', 16),
        'STATE': int('323', 16),
    },
    'ARD_ID': {
        int('201', 16)
    },
    'STEER': { # TODO LINK TO DATASHEET AND NAME
        'ID': int('601', 16),  # 0x601, 0x600 + number in switch of pcb
        
        # INIT MESSAGES
        # MSG A: Perfil de posición
        'MSG_A': [
            int('2F', 16), # Send 1 byte
            int('60', 16), # Index low
            int('60', 16), # Index high
            int('00', 16), # Subindex
            int('01', 16), # Data 0
        ],
        # MSG B: Parámetros
        'MSG_B': [
            int('00', 16),
            int('00', 16),
            int('00', 16),
            int('00', 16),
            int('00', 16),
            int('00', 16),
            int('00', 16),
            int('00', 16),
        ],
        # MSG C: Habilitar
        'MSG_C': [
            int('2B', 16), # Send 2 bytes
            int('40', 16), # Index low
            int('60', 16), # Index high
            int('00', 16), # Subindex
            int('0F', 16), # Data 0
            int('00', 16), # Data 1
        ],
        # MSG N: Deshabilitar
        'MSG_N': [
            int('2B', 16), # Send 2 bytes
            int('40', 16), # Index low
            int('60', 16), # Index high
            int('00', 16), # Subindex
            int('06', 16), # Data 0
            int('00', 16), # Data 1
        ],
        # LOOP MESSAGES
        # MSG F: Toggle New Position Bit TODO EXPLAIN BETTER and possibly action in dictionary
        'MSG_F': [
            int('2B', 16), # Send 2 bytes
            int('40', 16), # Index low
            int('60', 16), # Index high
            int('00', 16), # Subindex
            int('0F', 16), # Data 0
            int('00', 16), # Data 1
        ],
        # MSG D: Posicion objetivo
        'MSG_D': [
            int('23', 16),
            int('7A', 16),
            int('60', 16),
            int('00', 16),
        ],
        # MSG E: Orden de posicionamiento
        'MSG_E': [
            int('2B', 16), # Send 2 bytes
            int('40', 16), # Index low
            int('60', 16), # Index high
            int('00', 16), # Subindex
            int('3F', 16), # Data 0
            int('00', 16), # Data 1
        ],
    },
    'ASSIS': { # State lights of the car
        'COCKPIT': int('350', 16),
        'RIGHT': int('351', 16),
        'LEFT': int('352', 16),
    }
}