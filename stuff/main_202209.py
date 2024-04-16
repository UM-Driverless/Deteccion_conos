import cv2
import numpy as np
from connection_utils.my_client import ConnectionManager
from agent.agent_v0 import Agent
from cone_detection.detection_utils_v1 import ConeDetector
from trajectory_estimation.cone_processing import ConeProcessing
from visualization_utils.visualize import Visualize
import tensorflow as tf

# Estas tres lineas resuelven algunos problemas con cuDNN en TF2 por los que no me permitía ejecutar en GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == '__main__':
    verbose = 1
    # Inicializar detector de conos
    detector = ConeDetector()
    cone_processing = ConeProcessing()
    # Inicializar Agente (controlador)
    agent = Agent()
    # Inicializar conexiones
    connect_mng = ConnectionManager()
    # Visualización de datos
    visualizator = Visualize()

    try:
        while True:

            # Pedir datos al simulador o al coche
            image, speed, throttle, steer, brake = connect_mng.get_data(verbose=1)

            # Detectar conos
            detections, [cone_centers, eagle_img] = detector.detect_cones(image,
                                                                                 show_detections=False,
                                                                                 min_score_thresh=0.3,
                                                                                 real_time=True,
                                                                                 im_name='output')


            actions, ref_point, wrap_centers, order_wrap_centers = \
                agent.get_action(detections, cone_centers, eagle_img, image.shape)

            connect_mng.send_actions(throttle=actions[0], brake=actions[1], steer=actions[2])

            # Visualizar imagenes
            if verbose > 0:
                visualizator.show_cone_map(ref_point, wrap_centers[0], wrap_centers[1], order_wrap_centers[2],
                                           order_wrap_centers[3], image.shape)
                visualizator.draw_joined_cones(order_wrap_centers[0], order_wrap_centers[1], order_wrap_centers[2],
                                               order_wrap_centers[3], image.shape)
                visualizator.show_detections(detections, image, detector)

                cv2.waitKey(1)
                if cv2.waitKey(1) == ord('q'):
                    break

    finally:
        connect_mng.close_connection()
