from connection_utils.car_comunication import ConnectionManager
# from connection_utils.car_comunication import ConnectionManager_dummy as ConnectionManager
from controller_agent.testing_agent import AgentActuatorsTest as Agent
from visualization_utils.visualizer_test_actuators import VisualizeActuators as Visualizer
from visualization_utils.logger import Logger
import os
import time

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# # os.environ["CUDA_DEVICE_ORDER"] = '0'

if __name__ == '__main__':
    verbose = 1

    logger_path = os.path.join(os.getcwd(), "logs")
    init_message = "actuator_testing.py"
    logger = Logger(logger_path, init_message)
    # Inicializar detector de conos
    # detector = ConeDetector()

    # Inicializar conexiones
    connect_mng = ConnectionManager(logger)

    # Inicializar Agente (controlador)
    agent = Agent()

    # Visualización de datos
    visualizer = Visualizer(max_data_to_store=10000)

    try:
        while True:
            start_time = time.time()

            # Pedir datos al simulador o al coche
            image, in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear = connect_mng.get_data(verbose=1)

            # Detectar conos
            # detections, y_hat = detector.detect_cones(image)

            # Actions:
            # 1 -> steer
            # 2 -> throttle
            # 3 -> brake
            # 4 -> clutch
            # 5 -> upgear
            # 6 -> downgear

            # Seleccionar acciones
            throttle, brake, steer, clutch, gear = agent.get_action([1, 2, 3, 4])

            # resize actions
            throttle *= 0.8
            brake *= 0.8
            steer *= 0.8
            clutch *= 0.8

            # Enviar acciones
            connect_mng.send_actions(throttle=throttle,
                                     brake=brake,
                                     steer=steer,
                                     clutch=clutch,
                                     gear=gear)

            print("FPS: ", 1.0 / (time.time() - start_time))

            if verbose == 1:
                visualizer.visualize([in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear],
                                     [throttle, brake, steer, clutch, gear],
                                     print_can_data=True,
                                     print_agent_actions=True,
                                     real_time=True)

    finally:
        # Do whatever needed where the program ends or fails
        # connect_mng.close_connection()
        visualizer.visualize([in_speed, in_throttle, in_steer, in_brake, in_clutch, in_gear],
                             [throttle, brake, steer, clutch, gear],
                             print_can_data=True,
                             print_agent_actions=True,
                             real_time=False)

        visualizer.close_windows()