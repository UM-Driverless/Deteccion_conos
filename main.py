"""
MAIN script to run all the others. Shall contain the main classes, top level functions etc.


# HOW TO USE
- Configuration variables in globals.py
- If CAMERA_MODE = 4, start the simulator first, running fsds-v2.2.0-linux/FSDS.sh (This program can be located anywhere)
    - Make sure there's a camera called 'cam1'
    - Make sure the simulator folder is called 'Formula-Student-Driverless-Simulator'
- You may need in the home directory:
    - [Deteccion_conos](https://github.com/UM-Driverless/Deteccion_conos)
    - [Formula-Student-Driverless-Simulator](https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator)

- CHECK THIS:
    - Weights in yolov5/weights/yolov5_models
    - Active bash folder is ~/Deteccion_conos/
    - Check requirements{*}.txt, ZED API and gcc compiler up to date (12.1.0), etc.

- For CAN to work first run setup_can0.sh

# REFERENCES
https://github.com/UM-Driverless/Deteccion_conos/tree/Test_Portatil
vulture . --min-confidence 100

# TODO
- Solve TORNADO.PLATFORM.AUTO ERROR, WHEN USING SIMULATOR
- MAKE ZED WORK AGAIN
- RESTORE GENERIC AGENT CLASS FOR NO SPECIFIC TEST. THEN THE TESTS INHERIT FROM IT. COMMENTED.
- PUT GLOBAL VARS AS ATTRIBUTE OF CAR OBJECT?
- PROBAR CAN EN UM05
- IPYTHON TO REQUIREMENTS, also canlib
- Initialize trackbars of ConeProcessing. Why?
- Only import used libraries from activations with global config constants
- SET SPEED ACCORDING TO CAN PROTOCOL, and the rest of state variables (SEN BOARD)
- check edgeimpulse
- Print number of cones detected per color
- Xavier why network takes 3s to execute. How to make it use GPU?
- Make net faster. Remove cone types that we don't use? Reduce resolution of yolov5?
- Move threads to different files to make main.py shorter
- Check NVPMODEL with high power during xavier installation
- Reuse logger

- Wanted to make visualize work in a thread and for any resolution, but now it works for any resolution, don't know why, and it's always about 3ms so it's not worth it for now.

# STUFF
To stop: Ctrl+C in the terminal

# INFO
In ruben laptop:
    torch.hub.load(): YOLOv5 🚀 2023-1-31 Python-3.10.8 torch-1.13.0+cu117 CUDA:0 (NVIDIA GeForce GTX 1650, 3904MiB)
    torch.hub.load(): YOLOv5 🚀 2023-4-1 Python-3.10.6 torch-1.11.0+cu102 CUDA:0 (NVIDIA GeForce GTX 1650, 3904MiB)
- ORIN VERSIONS:
    - Jetpack 5.1.1 (installs CUDA11.4), pytorch 2.0.0+nv23.05 (for arm64 with cuda), torchvision version 0.15.1 with cuda, for that run this code:
        ```bash
        $ sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
        $ git clone --branch v0.15.1 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
        $ cd torchvision
        $ export BUILD_VERSION=0.15.1 # where 0.x.0 is the torchvision version  
        $ python3 setup.py install --user
        ```
    
- S_b: Body Coordinate system, origin in camera focal point:
    - X: Forwards
    - Y: Left
    - Z: Up

"""

if __name__ == '__main__': # multiprocessing creates child processes that import this file, with __name__ = '__mp_main__'
    # IMPORTS
    import os
    print(f'Current working directory: {os.getcwd()}') # The terminal should be in this directory
    
    import time
    import numpy as np
    import multiprocessing
    import matplotlib.pyplot as plt # For representation of time consumed
    import sys
    print(f'Python version: {sys.version}')
    
    from globals.globals import * # Global variables and constants, as if they were here

    if (CAN_MODE == 1):
        from connection_utils.can_kvaser import CanKvaser
    elif (CAN_MODE == 2):
        from connection_utils.can_xavier import CanJetson

    from cone_detection.yolo_detector import ConeDetector
    from visualization_utils.visualizer_yolo_det import Visualizer
    from visualization_utils.logger import Logger
    import cv2 # Webcam
    
    cam_queue  = multiprocessing.Queue(maxsize=1) #block=True, timeout=None. Global variable

    # INITIALIZE things
    ## Logger
    logger = Logger("Logger initialized")

    ## Cone detector
    detector = ConeDetector(checkpoint_path=WEIGHTS_PATH)

    # SIMULATOR
    if (CAMERA_MODE == 4):
        # With https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator cloned in the home directory:
        fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
        sys.path.insert(0, fsds_lib_path)
        print(f'FSDS simulator path: {fsds_lib_path}')
        import fsds # TODO why not recognized when debugging
        
        # connect to the simulator
        sim_client1 = fsds.FSDSClient() # To get the image
        sim_client2 = fsds.FSDSClient() # To control the car

        # Check network connection, exit if not connected
        sim_client1.confirmConnection()
        sim_client2.confirmConnection()
        
        # Control the Car
        sim_client2.enableApiControl(True) # Disconnects mouse control, only API with this code
        simulator_car_controls = fsds.CarControls()

    # SIMULATOR MANUAL -> CAR WONT MOVE ON ITS OWN
    if (CAMERA_MODE == 5):
        # With https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator cloned in the home directory:
        fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
        sys.path.insert(0, fsds_lib_path)
        print(f'FSDS simulator path: {fsds_lib_path}')
        import fsds # TODO why not recognized when debugging
        
        # connect to the simulator
        sim_client1 = fsds.FSDSClient() # To get the image
        sim_client2 = fsds.FSDSClient() # To control the car

        # Check network connection, exit if not connected
        sim_client1.confirmConnection()
        sim_client2.confirmConnection()
        
        # No control. Give to keyboard
        sim_client2.enableApiControl(False)
            
    # THREAD FUNCTIONS
    from thread_functions import *
    import cv2

    def can_send_thread(can_queue, can_receive):
        print(f'Starting CAN receive thread...')
        
        while True:
            can_receive.receive_frame() # can_receive.frame updated
            # print(f'FRAME RECEIVED: {can_receive.frame}')
            # global car_state
            car_state_local = can_receive.new_state(car_state)
            # print(car_state_local)
            can_queue.put(car_state_local)

    def read_image_zed(cam_queue):
        '''
        Read the ZED camera - https://www.stereolabs.com/docs/video/camera-controls/
        '''
        
        import cv2
    
        print(f'Starting read_image_webcam thread...')

        cam = cv2.VideoCapture(CAM_INDEX)
        
        # SETTINGS
        # cam.set(cv2.CAP_PROP_FPS, 60)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640*2) #1280 640 default
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #720  480 default
        # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 5% speed increase
        
        if (cam.isOpened() == False): 
            print("Error opening webcam")
        
        while True:
            # recorded_times_0 = time.time()
            
            # Read image from webcam
            # TODO also check CHECK cam.isOpened()?
            # It's 3 times faster if there are cones being detected. Nothing to do with visualize.
            result, image = cam.read()
            while result == False:
                result, image = cam.read()
            
            # recorded_times_1 = time.time()
            
            # cv2.imshow('image',image)
            # cv2.waitKey(1)
                        
            image = cv2.resize(image, (640*2,640), interpolation=cv2.INTER_AREA)
            image = np.array(image)[:,0:640,:]
            
            if (FLIP_IMAGE):
                image = cv2.flip(image, flipCode=1) # For testing purposes
            
            cam_queue.put(image)
            # print(f'Webcam read time: {recorded_times_1 - recorded_times_0}')
        
        
        '''
        import pyzed.sl as sl
        
        print(f'Starting read_image_zed thread...')
        
        zed = sl.Camera()
        
        # Camera settings
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 100) # We don't want blurry photos, we don't care about noise. The exposure time will still be adjusted automatically to compensate lighting conditions
        zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 8) # Maximum so it recognizes the color of the cones better. 0 to 8
        # cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 0) # Doesn't seem to make much difference. 0 to 8
        #cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1) # We have fixed gain, so this is automatic. % of camera framerate (period)
        
        # Init parameters: https://www.stereolabs.com/docs/api/structsl_1_1InitParameters.html
        zed_params = sl.InitParameters()
        # zed_params.camera_fps = 100 # Not necessary. By default does max fps
        
        # RESOLUTION: HD1080 (3840x1080), HD720 (1280x720), VGA (VGA=1344x376)
        # yolov5 uses 640x640. VGA is much faster, up to 100Hz
        zed_params.camera_resolution = sl.RESOLUTION.VGA
        
        #zed_params.sdk_gpu_id = -1 # Select which GPU to use. By default (-1) chooses most powerful NVidia
        
        status = zed.open(zed_params)
        
        while status != sl.ERROR_CODE.SUCCESS:
            print(f'ZED ERROR: {status}')
            time.sleep(0.5)
            status = zed.open(zed_params)
        print('SUCESSSSSSS, zed opened')
        runtime = sl.RuntimeParameters()
        runtime.enable_depth = False # Deactivates de depth map calculation. We don't need it.
        
        # Create an RGBA sl.Mat object
        mat_img = sl.Mat()
        
        while True:
            # Read ZED camera
            if (zed.grab(runtime) == sl.ERROR_CODE.SUCCESS): # Grab gets the new frame
                print('.')
                # recorded_times_0 = time.time()
                
                zed.retrieve_image(mat_img, sl.VIEW.LEFT) # Retrieve receives it and lets choose views and colormodes
                image = mat_img.get_data() # Creates np.array()
                cam_queue.put(image)
                
                # recorded_times_1 = time.time()
                # print(f'ZED read time: {recorded_times_1-recorded_times_0}')
        '''

    ## Connections
    if (CAN_MODE == 1):
        # CAN with Kvaser

        can_receive = CanKvaser()
        can_send = CanKvaser()
        can_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None TODO probably bad parameters, increase maxsize etc.
        
        can_send_worker = multiprocessing.Process(target=can_send_thread, args=(can_queue, can_receive,), daemon=False)
        can_send_worker.start()
        
        print('CAN (Kvaser) initialized')
    elif (CAN_MODE == 2):
        # CAN with Jetson
        can_send = CanJetson()
        can_send.send_message()
        print('CAN (python-can, Jetson) initialized')

    ## Agent selection
    if (MISSION_SELECTED == 0): # Generic
        from agent.agent import Agent
        agent = Agent()
    elif (MISSION_SELECTED == 1): # Acceleration
        from agent.agent_acceleration_mission import Acceleration_Mission
        agent = Acceleration_Mission()
    elif (MISSION_SELECTED == 2): # Skidpad
        from agent.agent_skidpad_mission import Skidpad_Mission
        agent = Skidpad_Mission()
    else: # The default Agent is the class of which other agents inherit from
        raise Exception(f'ERROR: WRONG MISSION_SELECTED VALUE. Got {MISSION_SELECTED} but expected int from 0 to 2')

    agent_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None
    agent_in_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None

    # SETUP CAMERA
    if (CAMERA_MODE == 0):   cam_worker = multiprocessing.Process(target=read_image_file,   args=(cam_queue,), daemon=False)
    elif (CAMERA_MODE == 1): cam_worker = multiprocessing.Process(target=read_image_video,  args=(cam_queue,), daemon=False)
    elif (CAMERA_MODE == 2): cam_worker = multiprocessing.Process(target=read_image_webcam, args=(cam_queue,), daemon=False)
    elif (CAMERA_MODE == 3): cam_worker = multiprocessing.Process(target=read_image_zed,    args=(cam_queue,), daemon=False)
    elif (CAMERA_MODE == 4): cam_worker = multiprocessing.Process(target=read_image_simulator, args=(cam_queue,sim_client1,), daemon=False)
    elif (CAMERA_MODE == 5): cam_worker = multiprocessing.Process(target=read_image_simulator, args=(cam_queue,sim_client1,), daemon=False)

    cam_worker.start()

    # READ TIMES
    TIMES_TO_MEASURE = 4
    recorded_times = np.array([0.]*(TIMES_TO_MEASURE+2)) # Timetags at different points in code
    integrated_time_taken = np.array([0.]*TIMES_TO_MEASURE)
    average_time_taken = np.array([0.]*TIMES_TO_MEASURE)
    fps = -1.
    integrated_fps = 0.
    loop_counter = 0

    ## Data visualization
    if (VISUALIZE == 1):
        visualizer = Visualizer()
        # visualize_worker = multiprocessing.Process(target=visualize_thread, args=(), daemon=False)

    # Main loop ------------------------
    try:
        print(f'Starting main loop...')
        while True:
            recorded_times[0] = time.time()

            image = cam_queue.get(timeout=4)
            # cv2.imshow('im',image)
            cv2.waitKey(1)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Resize to IMAGE_RESOLUTION no matter how we got the image
            
            # CROP
            # height = len(image)
            # width = len(image[0])
            # image = np.array(image)[height-640:height, int(width/2)-320:int(width/2)+320]
            
            image = cv2.resize(image, IMAGE_RESOLUTION, interpolation=cv2.INTER_AREA)
            
            # image = np.array(image)
            # Save to file (optional)
            # cv2.imwrite('image.png',image)
            
            recorded_times[1] = time.time()

            # Detect cones
            cones = detector.detect_cones(image)
            recorded_times[2] = time.time()
            
            # Update car values from CAN
            if (CAN_MODE == 1):
                car_state = can_queue.get() # TODO there are much more CAN things to do this.
            
            # Get actions from agent
            if (CAMERA_MODE == 4):
                agent.act_sim(
                                    cones,
                                    sim_client2 = sim_client2,
                                    simulator_car_controls = simulator_car_controls
                                    )                                                                                           
            else:
                agent.act(cones)

            recorded_times[3] = time.time()

            # Send actions - CAN
            if (CAN_MODE == 1):
                # Send target values from agent
                can_send.send_frame()
            
            # VISUALIZE
            # TODO add parameters to class
            if (VISUALIZE == 1):
                in_speed = 0
                in_rpm = 0

                visualizer.visualize(agent_target,
                                    car_state,
                                    image,
                                    cones,
                                    fps,
                                    save_frames=False)
            
            recorded_times[4] = time.time()

            # END OF LOOP
            loop_counter += 1
            fps = 1/(recorded_times[TIMES_TO_MEASURE] - recorded_times[0])
            integrated_fps += fps
            integrated_time_taken += np.array([(recorded_times[i+1]-recorded_times[i]) for i in range(TIMES_TO_MEASURE)])
            
    finally:
        # When main loop stops, due to no image, error, Ctrl+C on terminal, this calculates performance metrics and closes everything.

        # TIMES
        # cam.release()
        if loop_counter != 0:
            average_time_taken = integrated_time_taken/loop_counter
            fps = integrated_fps/loop_counter
            print(f'\n\n\n------------ RESULTS ------------\n',end='')
            print(f'FPS: {fps}')
            print(f'LOOPS: {loop_counter}')
            print(f'AVERAGE TIMES: {average_time_taken}')
            print(f'---------------------------------\n',end='')
            
            ## Plot the times
            fig = plt.figure(figsize=(12, 4))
            plt.bar(['cam.read()','detect_cones()','agent.act()','visualize'],average_time_taken)
            plt.ylabel("Average time taken [s]")
            plt.figtext(.8,.8,f'{fps:.2f}Hz')
            plt.title("Execution time per section of main loop")
            plt.savefig("logs/times.png")
        else:
            average_time_taken = -1
            fps = -1
            print("-------- ERROR, NO RESULTS --------")
        

        # Close processes and windows
        cam_worker.terminate()
        cv2.destroyAllWindows()
        
        agent_target = {
            "throttle": 0.,
            "brake": 0.,
            "steer": 0.,
            "clutch": 0.,
        }
        
        # Give sim control back
        if (CAMERA_MODE == 4):
            sim_client2.enableApiControl(False) # Allows mouse control, only API with this code
