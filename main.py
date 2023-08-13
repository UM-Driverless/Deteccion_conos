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
    - Active bash folder is ~/Deteccion_conos/ (Otherwise hubconf.py error)
    - Check requirements{*}.txt, ZED API and gcc compiler up to date (12.1.0), etc.

- For CAN to work first run setup_can0.sh
    - To run on startup, add to /etc/profile.d/

# REFERENCES
https://github.com/UM-Driverless/Deteccion_conos/tree/Test_Portatil
vulture . --min-confidence 100

# TODO
TODO with open to camera and threads, simulator control? So it can close when stopped.
- TODO RECOVER SIMULATOR CONTROL
- SEND CAN HEARTBEAT
- TODO THREADED CAN
- CAN in threads, steering with PDO instead of SDO (faster)
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
- In ruben laptop:
    - torch.hub.load(): YOLOv5 🚀 2023-1-31 Python-3.10.8 torch-1.13.0+cu117 CUDA:0 (NVIDIA GeForce GTX 1650, 3904MiB)
    - ( torch.hub.load(): YOLOv5 🚀 2023-4-1 Python-3.10.6 torch-1.11.0+cu102 CUDA:0 (NVIDIA GeForce GTX 1650, 3904MiB) )
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
    os.system('clear')
    print(f'Current working directory: {os.getcwd()}') # The terminal should be in this directory
    
    import time
    import numpy as np
    import multiprocessing
    import matplotlib.pyplot as plt # For representation of time consumed
    import sys
    from tools.time_counter import Time_Counter
    print(f'Python version: {sys.version}')
    
    from car import Car
    
    from globals.globals import * # Global variables and constants, as if they were here

    from visualization_utils.visualizer_yolo_det import Visualizer
    import cv2 # Webcam

    # INITIALIZE things
    dv_car = Car()
    
    if LOGGER:
        dv_car.logger.write(f'CONFIG: CAN_MODE = {CAN_MODE}, CAMERA_MODE = {CAMERA_MODE}, CAM_INDEX = {CAM_INDEX}, VISUALIZE = {VISUALIZE}, MISSION_SELECTED = {MISSION_SELECTED}')

    # READ TIMES
    timer = Time_Counter()

    if (MISSION_SELECTED == 6):
        VISUALIZE = 0

    ## Data visualization
    if (VISUALIZE == 1):
        visualizer = Visualizer()
        # visualize_worker = multiprocessing.Process(target=visualize_thread, args=(), daemon=False)
    

    # Main loop ------------------------
    try:        
        print(f'Starting main loop...')
        AS_Finished = False
        while (not AS_Finished):
            timer.add_time()

            dv_car.get_data()
            
            # cv2.imshow('im',dv_car.image)
            # cv2.waitKey(1)
            
            dv_car.image = cv2.cvtColor(dv_car.image, cv2.COLOR_RGB2BGR)
            # Resize to IMAGE_RESOLUTION no matter how we got the image
            
            # CROP
            # height = len(image)
            # width = len(image[0])
            # image = np.array(image)[height-640:height, int(width/2)-320:int(width/2)+320]
            
            dv_car.image = cv2.resize(dv_car.image, IMAGE_RESOLUTION, interpolation=cv2.INTER_AREA)
            
            timer.add_time()
            # Detect cones
            print(f'--------Loop counter:   {dv_car.loop_counter}--------')
            dv_car.cones = dv_car.detector.detect_cones(dv_car.image, dv_car.state)
            timer.add_time()
            AS_Finished = dv_car.calculate_actuation()
            timer.add_time()
            dv_car.send_actuation()
            # VISUALIZE
            # TODO add parameters to class
            timer.add_time()
            if (VISUALIZE == 1):
                in_speed = 0
                in_rpm = 0

                visualizer.visualize(dv_car.actuation,
                                    dv_car.state,
                                    dv_car.image,
                                    dv_car.cones,
                                    save_frames=False)

            if LOGGER:
                dv_car.logger.write(f'Iter {dv_car.loop_counter}, FPS: {dv_car.state["fps"]:.1f}, STATE: {dv_car.state}, ACTUATION: {dv_car.actuation}')
            # Save times
            timer.add_time()
            dv_car.state['fps'] = 1 / (timer.recorded_times[-1] - timer.recorded_times[0])
            if len(ifps)>20:
                ifps.pop(0)
            ifps.append(1 / (timer.recorded_times[-1] - timer.recorded_times[0]))
            timer.new_iter()
            dv_car.loop_counter += 1
            print()
    finally:
        print(f'------------')
        if LOGGER:
            dv_car.logger.write(f'{np.array(timer.times_integrated) / timer.loop_counter}')
        ## Plot the times
        fig = plt.figure(figsize=(12, 4))
        plt.bar(['get_data()','detect_cones()','calculate_actuation()','send_actuation'],np.array(timer.times_integrated) / timer.loop_counter)
        plt.ylabel("Average time taken [s]")
        plt.figtext(.8,.8,f'{1/(timer.times_integrated[-1] - timer.times_integrated[0]):.2f}Hz')
        plt.title("Execution time per section of main loop")
        plt.savefig("logs/times.png")
        
        # Close processes and windows
        dv_car.terminate()

