from connection_utils.car_comunication import ConnectionManager
from controller_agent.agent import AgentAccelerationYoloFast as AgentAcceleration
from controller_agent.agent_ebs import AgentTest180 as AgentTest
from cone_detection.yolo_detector import ConeDetector
from visualization_utils.visualizer_yolo_det import Visualizer
from visualization_utils.logger import Logger
from globals import can_constants
import os
import time
import cv2

import pyzed.sl as sl
import numpy as np


#######################################################################################################################
# Este código es más rápido por que usa AgentAccelerationYoloFast. Este agente no realiza una proyección de la imagen y
# por lo tanto no calcula un mapa de los conos. Resliza directamente los cáculos sobre la imagen en la perspectiva
# original. Esto lo hace más sensible a errores, pero más rápido
#######################################################################################################################

def inicio_de_movimiento():
    time.sleep(1)
    start_time = time.time()
    print('-----inicio mov 2.1-----')
    connect_mng.send_actions(throttle=0.3,
                             brake=0,
                             steer=0,
                             clutch=1.0,
                             upgear=0,
                             downgear=0)
    time.sleep(2)
    start_time = time.time()
    print('-----inicio mov 2.2-----')
    for x in range(100, 0, -5):
        connect_mng.send_actions(throttle=0.3,
                                 brake=0,
                                 steer=0,
                                 clutch=x / 100,
                                 upgear=0,
                                 downgear=0)
        time.sleep(0.05)
        start_time = time.time()

    print('-----inicio mov 2.3-----')
    connect_mng.send_actions(throttle=0.3,
                             brake=0,
                             steer=0,
                             clutch=0,
                             upgear=0,
                             downgear=0)

def seleccion_agente_arrancado(agente):
    # Inicializar Agente (controlador)
    if agente == 1:
        print('---------------ACCELERATION--------------')
        agent_init = AgentAcceleration(logger=logger, target_speed=20.)
    elif agente == 2:
        print('---------------TEST180--------------')
        agent_init = AgentTest(logger=logger, target_speed=5.)
    print('agent initialized')

    #connect_mng.do_read_msg()
    check = 0
    # comprobar que el motor esta encendido
    while check <= 3:
        connect_mng.send_actions(throttle=0,
                                 brake=0,
                                 steer=0,
                                 clutch=1.0,
                                 upgear=0,
                                 downgear=0)
        rpm = connect_mng.can.get_rpm_can()
        print(f"----RPM: {rpm}")
        rpm=2000
        if rpm < can_constants.RPM_RALENTI:
            time.sleep(0.5)
        else:
            check += 1
            print('-----CHECK----')
    print('-----coche arrancado-----')
    time.sleep(10)
    return agent_init

if __name__ == '__main__':
    verbose = 1
    PATH_OUTPUT = "nombre_video.avi"
    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    out = cv2.VideoWriter(PATH_OUTPUT, fourcc, 24.0, (1024, 768))

    cam = sl.Camera()
    cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 0)
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 30
    # init.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    # init.coordinate_units = sl.UNIT.METER
    # init.depth_minimum_distance = 0.15

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    runtime = sl.RuntimeParameters()
    # runtime.sensing_mode = sl.SENSING_MODE.FILL

    logger_path = os.path.join(os.getcwd(), "logs")
    init_message = "actuator_zed_testing.py"
    logger = Logger(logger_path, init_message)
    # Inicializar detector de conos
    detector = ConeDetector(logger=logger)
    print('Cone detector initialized')
    # Inicializar conexiones
    connect_mng = ConnectionManager(logger=logger)
    print('CAN connection initialized')
    agent_run = False
    # Visualización de datos
    visualizer = Visualizer()
    print('visualizer initialized')
    mat_img = sl.Mat()
    accel_init = 0
    amr = 2
    try:
        while True:
            # Pedir datos al simulador o al coche
            #in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(verbose=0)
            # 1. Comprobar en que mision estamos (en este caso aceleración) 0x410-0
            # manual = 0, acc = 1, skidpad = 2, autox = 3, track = 4, ebstest= 5, inspection = 6
            print("-----Lectura amr-----")
            #amr = connect_mng.can.get_amr()

            if amr > 0:
                if accel_init == 0:
                    agent = seleccion_agente_arrancado(amr)
                    inicio_mov = 0
                    accel_init = 1

                start_time = time.time()
                # Pedir datos al simulador o al coche
                # in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(verbose=0)
                time_send = time.time()
                # se obtienen las detecciones
                err = cam.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(mat_img)
                    image = mat_img.get_data()

                    np.array(image)
                    # cv2.imshow("img", image)
                    # cv2.waitKey(1)
                    # in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(
                    #     verbose=0)
                    # Detectar conos
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    out.write(image)
                    print("---get_image---")
                    print(time.time() - time_send)

                    # in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(
                    #     verbose=0)
                    time_send = time.time()
                    detections, cone_centers = detector.detect_cones(image, get_centers=True)
                    print("---get_detections---")
                    print(time.time() - time_send)

                    time_send = time.time()
                    in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(
                        verbose=0)
                    print("---get_data---")
                    print (time.time() - time_send)
                    # in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = [0, 0, 0, 0, 0, 0, 0]
                    # Actions:
                    # 1 -> steer
                    # 2 -> throttle
                    # 3 -> brake
                    # 4 -> clutch
                    # 5 -> upgear
                    # 6 -> downgear

                    # Inicio de movimiento

                    if inicio_mov == 0:
                        inicio_de_movimiento()
                        inicio_mov = 1
                    time_send = time.time()
                    # Seleccionar acciones
                    [throttle, brake, steer, clutch, upgear, downgear, gear], data = agent.get_action(
                        detections=detections,
                        speed=in_speed,
                        gear=in_gear,
                        rpm=in_rpm,
                        cone_centers=cone_centers,
                        image=image)
                    print("---get_action---")
                    print (time.time() - time_send)
                    # resize actions
                    # throttle *= 0.8
                    # brake *= 0.8
                    # steer *= 0.8
                    # clutch *= 0.8
                    time_send = time.time()
                    # Enviar acciones
                    connect_mng.send_actions(throttle=0.3,
                                             brake=brake,
                                             steer=steer,
                                             clutch=clutch,
                                             upgear=upgear,
                                             downgear=downgear)
                    print("---send_action---")
                    print (time.time() - time_send)
                    # in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(
                    #     verbose=0)
                    # print(in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm)
                    fps = 1.0 / (time.time() - start_time)
                    # print("FPS: ", 1.0 / (time.time() - start_time))

                if verbose == 1:
                    time_send = time.time()
                    cenital_map = [data[1], data[2], data[-1]]
                    visualizer.visualize([image, detections, cone_centers, cenital_map, in_speed],
                                         [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm], fps,
                                         save_frames=True)
                    print("-----Visualizacion-----)")
                    print(time.time() - time_send)
                    # in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(
                    #     verbose=0)



            elif amr == 0:
                accel_init = 0
                # .... resetear todos los init



    finally:
        throttle = 0
        brake = 0
        steer = 0
        clutch = 0
        upgear = 0
        downgear = 0
        connect_mng.send_actions(throttle=throttle,
                                 brake=brake,
                                 steer=steer,
                                 clutch=clutch,
                                 upgear=upgear,
                                 downgear=downgear)
        out.release()
        print("fin")
        # visualizer.save_in_video(path = "videos", name = "avi")
        # visualizer.close_windows()
