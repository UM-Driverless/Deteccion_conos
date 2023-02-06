from connection_utils.my_client import ConnectionManager
from agent.agent import AgentAccelerationYolo as AgentAcceleration
from cone_detection.yolo_detector import ConeDetector
from visualization_utils.visualizer_yolo_det import Visualizer
from visualization_utils.logger import Logger
import os
import time

# os.environ["CUDA_DEVICE_ORDER"] = '0'

#######################################################################################################################
# Este código es más lento pero más fiable por que usa AgentAccelerationYolo. Este agente realiza una proyección de la
# imagen y por lo tanto calcula un mapa de los conos. A partir de este mapa realiza los cáculos.
#######################################################################################################################

if __name__ == '__main__':

    verbose = 1

    logger_path = os.path.join(os.getcwd(), "logs")
    init_message = "sim_acceleration.py"
    logger = Logger(logger_path, init_message)

    # Inicializar detector de conos
    detector = ConeDetector(logger=logger)

    # Inicializar conexiones
    connect_mng = ConnectionManager(logger=logger)

    # Inicializar Agente (controlador)
    agent = AgentAcceleration(logger=logger, target_speed=60.)
    # Visualización de datos
    visualizer = Visualizer()

    count = 0
    try:
        while True:
            start_time = time.time()
            # Pedir datos al simulador o al coche
            image, in_speed, in_throttle, in_steer, in_brake, in_gear, in_rpm = connect_mng.get_data(verbose=0)

            # Detectar conos
            detections, cone_centers = detector.detect_cones(image, get_centers=True)

            [throttle, brake, steer, clutch, upgear, downgear, gear], data = agent.get_action(detections=detections, speed=in_speed, gear=in_gear, rpm=in_rpm, cone_centers=cone_centers, image=image)

            connect_mng.send_actions(throttle=throttle, brake=brake, steer=steer, clutch=clutch, upgear=upgear, downgear=downgear)

            fps = 1.0 / (time.time() - start_time)

            if verbose==1:
                cenital_map = [data[1], data[2], data[-1]]
                visualizer.visualize([image, detections, cone_centers, cenital_map, in_speed], [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm], fps, save_frames=False)

            count += 1
            if in_speed <= 0. and count > 100:
                break


    finally:
        path = '/media/archivos/UMotorsport/video_yolo/'
        name = 'video_{:0>4d}.jpg'
        visualizer.save_in_video(path, name)
        connect_mng.close_connection()
