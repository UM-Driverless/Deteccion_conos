from connection_utils.my_client import ConnectionManager
from controller_agent.agent import Agent
from cone_detection.cone_segmentation import ConeDetector
from visualization_utils.visualizer_con_det_seg import Visualizer
import tensorflow as tf
import os
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# os.environ["CUDA_DEVICE_ORDER"] = '0'

if __name__ == '__main__':
    verbose = 1

    # Inicializar detector de conos
    detector = ConeDetector()

    # Inicializar conexiones
    connect_mng = ConnectionManager()

    ## Controlar con joystick (o ratón)
    # Inicializar Agente (controlador)
    # agent = Agent(connect_mng.imageWidth, connect_mng.imageHeigth)
    # Visualización de datos
    # visualizer = agent.visualizer

    ## Controlador autónomo
    # Inicializar Agente (controlador)
    agent = Agent()
    # Visualización de datos
    visualizer = Visualizer()

    try:
        while True:
            start_time = time.time()
            # Pedir datos al simulador o al coche
            image, speed, throttle, steer, brake = connect_mng.get_data(verbose=1)

            # Detectar conos
            detections, y_hat = detector.detect_cones(image, bbox=False, centers=True)

            [throttle, brake, steer, clutch, upgear, downgear], data = agent.get_action(detections, y_hat)
            print(steer)
            connect_mng.send_actions(throttle=throttle, brake=brake, steer=steer)

            print("FPS: ", 1.0 / (time.time() - start_time))

            if verbose==1:
                cenital_map = [data[1], data[2]]
                visualizer.visualize([image, detections, cenital_map, y_hat], [throttle, brake, steer, clutch, upgear, downgear], save_frames=True)


    finally:
        path = '/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/video/'
        name = 'video_{:0>4d}.jpg'
        visualizer.save_in_video(path, name)
        connect_mng.close_connection()
