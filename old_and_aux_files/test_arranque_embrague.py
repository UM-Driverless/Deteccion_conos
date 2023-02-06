# from connection_utils.my_client import ConnectionManager
from connection_utils.car_comunication import ConnectionManager_dummy as ConnectionManager
from agent.agent import AgentTestClutchThrottle
from cone_detection.cone_segmentation import ConeDetector
from visualization_utils.visualizer_test_actuators import VisualizeActuators as Visualizer
from visualization_utils.logger import Logger
import tensorflow as tf
import os
import time

"""
Este código ejecuta un test para probar el arranque del coche manejando el embrague y el accelerador. Dirección centrada.
El objetivo de este script es ajustar los valores de embrague y acceleración al comportamiento real del coche. 
Lo que debe ocurrir es que el coche meta 1 marcha accelere, se mueva y frene inmediatamente."""

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# os.environ["CUDA_DEVICE_ORDER"] = '0'

if __name__ == '__main__':
    verbose = 1

    logger_path = os.path.join(os.getcwd(), "logs")
    init_message = "test_arranque_embrague.py"
    logger = Logger(logger_path, init_message)

    # Inicializar detector de conos
    # detector = ConeDetector()

    # Inicializar conexiones
    connect_mng = ConnectionManager(logger)

    # Inicializar Agente (controlador)
    agent = AgentTestClutchThrottle(logger=logger)
    # Visualización de datos
    visualizer = Visualizer()

    try:
        while True:
            start_time = time.time()
            # Pedir datos al simulador o al coche
            image, in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm = connect_mng.get_data(verbose=1)

            # Detectar conos
            # detections, y_hat = detector.detect_cones(image, bbox=False, centers=True)
            detections = None
            y_hat = None
            [throttle, brake, steer, clutch, upgear, downgear, gear], data = agent.get_action(detections, y_hat, in_speed, in_gear, in_rpm)
            connect_mng.send_actions(throttle=throttle, brake=brake, steer=steer, clutch=clutch, upgear=upgear, downgear=downgear)

            fps = 1.0 / (time.time() - start_time)

            if verbose==1:
                visualizer.visualize([in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear, in_rpm],
                                     [throttle, brake, steer, clutch, gear, in_rpm], real_time=True)

    finally:
        # path = '/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/video/'
        # name = 'video_{:0>4d}.jpg'
        # visualizer.save_in_video(path, name)
        pass

