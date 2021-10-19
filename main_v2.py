from connection_utils.my_client import ConnectionManager
from controller_agent.mouse_controller import Agent
from cone_detection.cone_segmentation import ConeDetector
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

    # Inicializar Agente (controlador)
    agent = Agent(connect_mng.imageWidth, connect_mng.imageHeigth)

    # Visualización de datos
    visualizer = agent.visualizer

    try:
        while True:
            start_time = time.time()
            # Pedir datos al simulador o al coche
            image, speed, throttle, steer, brake = connect_mng.get_data(verbose=1)

            # Detectar conos
            detections, y_hat = detector.detect_cones(image)

            throttle, brake, steer, clutch, upgear, downgear = agent.get_action()
            connect_mng.send_actions(throttle=throttle, brake=brake, steer=steer)

            print("FPS: ", 1.0 / (time.time() - start_time))

            visualizer.visualize([image, detections, y_hat])


    finally:
        # path = '/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/video/'
        # name = 'video_{:0>4d}.jpg'
        # visualizer.save_in_video(path, name)
        connect_mng.close_connection()
