import cv2
from controller_agent.agent import AgentAccelerationYoloFast as AgentAcceleration
from cone_detection.yolo_detector import ConeDetector
from visualization_utils.visualizer_yolo_det import Visualizer
from visualization_utils.logger import Logger

import os
import time

#######################################################################################################################
# Este código es más rápido por que usa AgentAccelerationYoloFast. Este agente no realiza una proyección de la imagen y
# por lo tanto no calcula un mapa de los conos. Resliza directamente los cáculos sobre la imagen en la perspectiva
# original. Esto lo hace más sensible a errores, pero más rápido
#######################################################################################################################
if __name__ == '__main__':
    image_folder = '/media/archivos/UMotorsport/ImagenesDataset/21_11_21/21_11_21/21_11_21_16_59_1/frames/'
    video_path = '/media/archivos/UMotorsport/aux/'
    video_name = 'video_{:0>4d}.jpg'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key= lambda x:int(x.split('-')[1].split('.')[0]))
    # images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = frame.shape

    verbose = 1
    logger_path = os.path.join(os.getcwd(), "logs")
    init_message = "sim_acceleration.py"
    logger = Logger(logger_path, init_message)

    # Inicializar detector de conos
    detector = ConeDetector(logger=logger)

    # Inicializar Agente (controlador)
    agent = AgentAcceleration(logger=logger, target_speed=60.)
    # Visualización de datos
    visualizer = Visualizer()

    images = [cv2.cvtColor(cv2.imread(os.path.join(image_folder, im)), cv2.COLOR_BGR2RGB) for im in images]

    try:
        for im in images:
            start_time = time.time()
            # Pedir datos al simulador o al coche
            frame = im
            # frame = cv2.imread(os.path.join(image_folder, im))
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            in_speed = 0.
            in_throttle = 0.
            in_steer = 0.
            in_brake = 0.
            in_gear = 0
            in_rpm = 0

            # Detectar conos
            detections, cone_centers = detector.detect_cones(frame, get_centers=True)

            [throttle, brake, steer, clutch, upgear, downgear, gear], data = agent.get_action(detections=detections, speed=in_speed, gear=in_gear, rpm=in_rpm, cone_centers=cone_centers, image=frame)

            fps = 1.0 / (time.time() - start_time)

            if verbose==1:
                cenital_map = [data[1], data[2], data[-1]]
                visualizer.visualize([frame, detections, cone_centers, cenital_map, in_speed], [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm], fps, save_frames=True)


    finally:
        ...
        # visualizer.save_in_video(video_path, video_name)
