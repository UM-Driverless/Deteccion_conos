import cv2

from globals.globals import IMAGE_RESOLUTION
from connection_utils.my_client import ConnectionManager
from agent.agent import AgentAccelerationYolo as AgentAcceleration
from cone_detection.yolo_detector import ConeDetector
from visualization_utils.visualizer_yolo_det import Visualizer
from visualization_utils.logger import Logger
import tensorflow as tf
import os
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# os.environ["CUDA_DEVICE_ORDER"] = '0'

if __name__ == '__main__':
    verbose = 1

    logger_path = os.path.join(os.getcwd(), "logs")
    init_message = "sim_acceleration.py"
    logger = Logger(logger_path, init_message)

    # Inicializar detector de conos
    detector = ConeDetector(logger=logger, checkpoint_path='yolov5/weights/yolov5_models/best_adri.pt')

    # Inicializar conexiones
    connect_mng = ConnectionManager(logger=logger)

    # Inicializar Agente (controlador)
    agent = AgentAcceleration(logger=logger, target_speed=60.)
    # Visualización de datos
    visualizer = Visualizer()

    try:
        while True:
            start_time = time.time()
            # Pedir datos al simulador o al coche
            image, speed, throttle, steer, brake, gear, rpm = connect_mng.get_data(verbose=0)
            car_state = {'speed': speed, 'throttle': throttle, 'steer': steer, 'brake': brake, 'gear': gear, 'rpm': rpm}
            image = cv2.resize(image, IMAGE_RESOLUTION, interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Detectar conos
            detections, cone_centers = detector.detect_cones(image, get_centers=True)

            agent_target, data = agent.get_action(detections=detections, car_state=car_state, cone_centers=cone_centers, image=image)

            connect_mng.send_actions(throttle=agent_target['throttle'], brake=agent_target['brake'], steer=agent_target['steer'], clutch=agent_target['clutch'], upgear=0, downgear=0)

            fps = 1.0 / (time.time() - start_time)

            if verbose == 1:
                cenital_map = [data[1], data[2]]
                visualizer.visualize(agent_target, car_state, image, detections, fps, save_frames=False
                                     )


    finally:
        # path = '/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/video/'
        # name = 'video_{:0>4d}.jpg'
        # visualizer.save_in_video(path, name)
        connect_mng.close_connection()
