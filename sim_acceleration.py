from connection_utils.my_client import ConnectionManager
from controller_agent.agent import AgentAcceleration
from cone_detection.cone_segmentation import ConeDetector
from visualization_utils.visualizer_con_det_seg import Visualizer
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
    detector = ConeDetector(logger=logger)

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
            image, in_speed, in_throttle, in_steer, in_brake, in_gear, in_rpm = connect_mng.get_data(verbose=0)

            # Detectar conos
            detections, y_hat = detector.detect_cones(image, bbox=False, centers=True)

            [throttle, brake, steer, clutch, upgear, downgear], data = agent.get_action(detections, y_hat, in_speed, in_gear, in_rpm)
            connect_mng.send_actions(throttle=throttle, brake=brake, steer=steer, clutch=clutch, upgear=upgear, downgear=downgear)

            fps = 1.0 / (time.time() - start_time)

            if verbose==1:
                cenital_map = [data[1], data[2]]
                visualizer.visualize([image, detections, cenital_map, y_hat, in_speed], [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm], fps, save_frames=False)


    finally:
        # path = '/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/video/'
        # name = 'video_{:0>4d}.jpg'
        # visualizer.save_in_video(path, name)
        connect_mng.close_connection()
