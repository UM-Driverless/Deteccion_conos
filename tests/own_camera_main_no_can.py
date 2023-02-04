from connection_utils.car_comunication import ConnectionManager
from controller_agent.agent import AgentAccelerationYolo as AgentAcceleration
from cone_detection.yolo_detector import ConeDetector
from visualization_utils.visualizer_yolo_det import Visualizer
from visualization_utils.logger import Logger
import os
import time
import cv2

import numpy as np

#######################################################################################################################
# Este código es más rápido por que usa AgentAccelerationYoloFast. Este agente no realiza una proyección de la imagen y
# por lo tanto no calcula un mapa de los conos. Resliza directamente los cáculos sobre la imagen en la perspectiva
# original. Esto lo hace más sensible a errores, pero más rápido
#######################################################################################################################

if __name__ == '__main__':
    verbose = 1


    # init.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    # init.coordinate_units = sl.UNIT.METER
    # init.depth_minimum_distance = 0.15

    # runtime.sensing_mode = sl.SENSING_MODE.FILL
    cam = cv2.VideoCapture(0)

    logger_path = os.path.join(os.getcwd(), "logs")
    init_message = "actuator_zed_testing.py"
    logger = Logger(logger_path, init_message)
    # Inicializar detector de conos
    detector = ConeDetector(logger=logger)
    print('Cone detector initialized')
    # Inicializar conexiones
    #connect_mng = ConnectionManager(logger=logger)
    print('CAN connection initialized')
    # Inicializar Agente (controlador)
    agent = AgentAcceleration(logger=logger, target_speed=60.)
    print('agent initialized')
    # Visualización de datos
    visualizer = Visualizer()
    print('visualizer initialized')
    try:
        while True:
            start_time = time.time()

            # Pedir datos al simulador o al coche
            #in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(verbose=1)

            result, image = cam.read()
            if result:
                np.array(image)
                # cv2.imshow("img", image)
                # cv2.waitKey(1)

                # Detectar conos
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                detections, cone_centers = detector.detect_cones(image, get_centers=True)

                # Actions:
                # 1 -> steer
                # 2 -> throttle
                # 3 -> brake
                # 4 -> clutch
                # 5 -> upgear
                # 6 -> downgear

                # Seleccionar acciones
                [throttle, brake, steer, clutch, upgear, downgear, gear], data = agent.get_action(detections=detections,
                                                                                                  speed=0,
                                                                                                  gear=0,
                                                                                                  rpm=0,
                                                                                                  cone_centers=cone_centers,
                                                                                                  image=image)

                # resize actions
                # throttle *= 0.8
                # brake *= 0.8
                # steer *= 0.8
                # clutch *= 0.8

                # Enviar acciones
                '''connect_mng.send_actions(throttle=throttle,
                                         brake=brake,
                                         steer=steer,
                                         clutch=clutch,
                                         upgear=upgear,
                                         downgear=downgear)'''

                fps = 1.0 / (time.time() - start_time)
                print("FPS: ", 1.0 / (time.time() - start_time))

            if verbose == 1:
                cenital_map = [data[1], data[2], data[-1]]
                in_speed = 0
                in_gear = 0
                in_rpm = 0

                visualizer.visualize([image, detections, cone_centers,cenital_map, in_speed],
                                     [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm], fps,
                                     save_frames=False)

    finally:
        throttle = 0
        brake = 0
        steer = 0
        clutch = 0
        upgear = 0
        downgear = 0

        visualizer.close_windows()
