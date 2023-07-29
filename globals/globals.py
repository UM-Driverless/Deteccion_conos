ifps = [-1]
maxSpeed = 7
tauP = 0.4#    0.6*Ku    0.4    0.2    0.09
tauD = 2.2#    2*Kp/Pu    2.4   3.0     1.40625
tauI = 0.004#    Kp*Pu/8    0.009   0.004   0.00144
lastAngle = [0]


# CONSTANTS FOR SETTINGS
CAN_MODE = 0 # 0 -> CAN OFF, default values to test without CAN, 1 -> Jetson (Embedded, socketcan python-can), 2 -> KVaser
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

# Actual state values of the car, from sensors
car_state = {
    'speed': -1., # m/s, can be calculated from speed_senfl mixed with other sensors.
    'rpm': -1.,
    'speed_senfl': -1., # speed according to SEN front left sensor
    'fps': -1.0,
    
}

# Target values obtained from agent.get_action()
agent_act = {
    'acc': 0., # Acceleration. From -1.0 to 1.0. TODO Then translates into throttle and brake
    'steer': 0., # -1.0 to 1.0
    'throttle': 0., # float in [0., 1.)
    'brake': 0., # float in [0., 1.)
}

# Neural net detections [{bounding boxes},{labels}] = [ [[[x1,y1],[x2,y2]], ...], [[{class name}, {confidence}], ...] ]
detections = []


# CAN
CAN_SEND_TIMEOUT = 0.005
CAN_ACTION_DIMENSION = 100.
MAXON_TOTAL_INCREMENTS = 122880. # Increments of maxon motor, from center, to get to the mechanical limit of the steering system at one side.

CAN_IDS = {
    'SENFL': { # Front Left wheel
        'IMU': 0x300, # 8 bytes - [Ax Ay Az Gx Gy Gz Mx My] TODO WRITE ID, its an id
        'SIG': 0x301, # 8 bytes - [Analog1 Analog2 Analog3 Digital Rel1 Rel2 Rel3]
        'STATE': 0x302, # 8 bytes - [Error id Null Null Null Null Null Null Null]
    },
    'SENFR': { # Front Right wheel
        'IMU': 0x303,
        'SIG': 0x304,
        'STATE': 0x305,
    },
    'SENRL': { # Rear Left wheel (UNUSED)
        'IMU': 0x306,
        'SIG': 0x307,
        'STATE': 0x308,
    },
    'SENRR': { # Rear Right wheel (UNUSED)
        'IMU': 0x309,
        'SIG': 0x310,
        'STATE': 0x311,
    },
    'ARD_ID': {
        0x201,
    },
    'STEER': { # TODO LINK TO DATASHEET AND NAME
        'MAXON_ID': 0x601,  # 0x601, 0x600 + number in switch of pcb
        'MAXON_HEARTBEAT_ID': 0x701,
    },
    'ASSIS': { # State lights of the car
        'COCKPIT': 0x350,
        'RIGHT': 0x351,
        'LEFT': 0x352,
    },
}

CAN_MSG = {
    'STEER': {
        
        # [EPOS4-Application-Notes-Collection-En_GOOD_OLD](https://www.maxongroup.com/medias/sys_master/root/8837359304734/EPOS4-Application-Notes-Collection-En.pdf)
        
        # # INIT
        # 1. DISABLE_POWER - cansend can0 601#2B40600600
        # 2. PROFILE_POSITION - cansend can0 601#2F60600001
        # 3. [OPTIONAL] SET_PARAMETERS - cansend can0 601#2360600001000000
        # 4. ENABLE_POWER - cansend can0 601#2B40600F00

        # # LOOP
        # 1. SET_TARGET_POS - cansend can0 601#237A600000E00100
        # 2. MOVE_ABSOLUTE_POS - cansend can0 601#2B4060003F00
        # 3. TOGGLE_NEW_POS - cansend can0 601#284060000F00
        # '''
        
        # '''Disable power - cansend can0 601#2B40600600
        # - ID = 601: SDO to MAXON PCB (11-bit standard id)
        # - 0x2B: Send 2 bytes of data (0x23: 4 bytes, 0x2B: 2 bytes, 0x2F: 1 bytes)
        # - 0x40 0x60 0x00: Controlword 0x6040-00.
        # - 0x06 0x00: Data 0x0006, Disable (0x000F = Enable)
        
        'DISABLE_POWER': [
            0x2B,
            0x40,
            0x60,
            0x00,
            0x06, # Least significant bit
            0x00, # Most significant bit
        ],
        
        # Configure to Profile Position - cansend can0 601#2F60600001
        # - ID = 601: SDO to MAXON PCB (11-bit standard id)
        # - 0x2F: Send 1 byte of data (0x23: 4 bytes, 0x2B: 2 bytes, 0x2F: 1 bytes)
        # - 0x60 0x60 0x00: Controlword 0x6060-00 -> Set the var of object dict: "Mode of operation"
        # - 0x01: Position Mode. (0x03 = Profile Position Mode)
        'PROFILE_POSITION': [
            0x2F,
            0x60,
            0x60,
            0x00,
            0x01,
        ],
        
        # Set parameters - OPTIONAL - Max velocity, acceleration etc. - cansend can0 601#2360600001000000
        # - ID = 601: SDO to MAXON PCB (11-bit standard id)
        # - 0x23: Send 4 bytes of data (0x23: 4 bytes, 0x2B: 2 bytes, 0x2F: 1 bytes)
        # - 0x60 0x60 0x00: Controlword 0x6060-00
        # - 0x01 0x00 0x00 0x00: Data, THESE ARE THE PARAMETERS: 0x00000001 (LSB first)
        'SET_PARAMETERS': [
            0x23,
            0x60,
            0x60,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
        ],
        
        # Enable power - cansend can0 601#2B40600F00
        # - ID = 601: SDO to MAXON PCB (11-bit standard id)
        # - 0x2B: Send 2 bytes of data (0x23: 4 bytes, 0x2B: 2 bytes, 0x2F: 1 bytes)
        # - 0x40 0x60 0x00: Controlword 0x6040-00
        # - 0x0F 0x00: Data 0x000F, Enable (0x0006: Disable)
        'ENABLE_POWER': [
            0x2B,
            0x40,
            0x60,
            0x00,
            0x0F,
            0x00,
        ],
        
        # SET_TARGET_POS - cansend can0 601#237A600000E00100
        # Add the 4 missing bytes depending on the desired position
        # - ID = 601: SDO to MAXON PCB (11-bit standard id)
        # - 0x23: Send 2 bytes of data (0x23: 4 bytes, 0x2B: 2 bytes, 0x2F: 1 bytes)
        # - 0x7A 0x60 0x00: Controlword 0x607A-00
        # - 0xab 0xcd 0xef 0xgh: 0xghefcdab increments
        'SET_TARGET_POS': [
            0x23,
            0x7A,
            0x60,
            0x00,
        ],
        
        # MOVE_ABSOLUTE_POS - cansend can0 601#2B4060003F00
        # - ID = 601: SDO to MAXON PCB (11-bit standard id)
        # - 0x2B: Send 2 bytes of data (0x23: 4 bytes, 0x2B: 2 bytes, 0x2F: 1 bytes)
        # - 0x40 0x60 0x00: Controlword 0x6040-00.
        # - 0x3F 0x00: Data 0x003F, Absolute Position Start Immediately (0x007F for Relative Position Start Immediately)
        'MOVE_ABSOLUTE_POS': [
            0x2B,
            0x40,
            0x60,
            0x00,
            0x3F,
            0x00,
        ],
        
        # MOVE_RELATIVE_POS - cansend can0 601#2B4060007F00
        # - ID = 601: SDO to MAXON PCB (11-bit standard id)
        # - 0x2B: Send 2 bytes of data (0x23: 4 bytes, 0x2B: 2 bytes, 0x2F: 1 bytes)
        # - 0x40 0x60 0x00: Controlword 0x6040-00.
        # - 0x7F 0x00: Data 0x007F, Relative Position Start Immediately (0x003F for Absolute Position Start Immediately)
        'MOVE_RELATIVE_POS': [
            0x2B,
            0x40,
            0x60,
            0x00,
            0x7F,
            0x00,
        ],
        
        # TOGGLE_NEW_POS - cansend can0 601#284060000F00
        # - ID = 601: SDO to MAXON PCB (11-bit standard id)
        # - 0x2B: Send 2 bytes of data (0x23: 4 bytes, 0x2B: 2 bytes, 0x2F: 1 bytes)
        # - 0x40 0x60 0x00: Controlword 0x6040-00.
        # - 0x0F 0x00: Data 0x000F, Enable (0x0006: Disable)
        'TOGGLE_NEW_POS': [
            0x2B,
            0x40,
            0x60,
            0x00,
            0x0F,
            0x00,
        ],
        
        
    },
}