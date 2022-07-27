from connection_utils.my_client import ConnectionManager
#from controller_agent.agent import AgentAccelerationYoloFast as AgentAcceleration
from controller_agent.agent_ebs import AgentTest180 as AgentAcceleration
from cone_detection.yolo_detector import ConeDetector
from visualization_utils.visualizer_yolo_det import Visualizer
from visualization_utils.logger import Logger
import os
import time

# os.environ["CUDA_DEVICE_ORDER"] = '0'

#######################################################################################################################
# Este código es más rápido por que usa AgentAccelerationYoloFast. Este agente no realiza una proyección de la imagen y
# por lo tanto no calcula un mapa de los conos. Resliza directamente los cáculos sobre la imagen en la perspectiva
# original. Esto lo hace más sensible a errores, pero más rápido
#######################################################################################################################

if __name__ == '__main__':
    verbose = 1

    logger_path = os.path.join(os.getcwd(), "logs")
    init_message = "sim_acceleration.py"
    logger = Logger(logger_path, init_message)

    # Inicializar detector de conos
    detector = ConeDetector(logger=logger, checkpoint_path=
    "pesos/yolov5_models/240.pt")

    # Inicializar conexiones
    connect_mng = ConnectionManager(logger=logger)

    # Inicializar Agente (controlador)
    agent = AgentAcceleration(logger=logger, target_speed=10.)
    # Visualización de datos
    visualizer = Visualizer()

    count = 0
    try:
        while True:
            start_time = time.time()
            # Pedir datos al simulador o al coche
            image, in_speed, in_throttle, in_steer, in_brake, in_gear, in_rpm = connect_mng.get_data(verbose=0)
            print(f"get_data: {time.time()-start_time}")
            # Detectar conos
            start_time1 = time.time()
            detections, cone_centers = detector.detect_cones(image, get_centers=True)
            print(f"detect_cones: {time.time() - start_time1}")

            start_time2 = time.time()
            [throttle, brake, steer, clutch, upgear, downgear, gear], data = agent.get_action(detections=detections,
                                                                                              speed=in_speed,
                                                                                              gear=in_gear, rpm=in_rpm,
                                                                                              cone_centers=cone_centers,
                                                                                              image=image)
            print(f"get_action: {time.time() - start_time2}")

            start_time3 = time.time()
            connect_mng.send_actions(throttle=throttle, brake=brake, steer=steer, clutch=clutch, upgear=upgear,
                                     downgear=downgear)
            print(f"send_actions: {time.time() - start_time3}")

            fps = 1.0 / (time.time() - start_time)
            start_time4 = time.time()
            if verbose == 1:
                cenital_map = [data[1], data[2], data[-1]]
                visualizer.visualize([image, detections, cone_centers, cenital_map, in_speed],
                                     [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm], fps,
                                     save_frames=False)
            print(f"visualize: {time.time() - start_time4}")

            print(fps)
            count += 1
            if in_speed <= 0. and count > 100:
                break


    finally:
        path = '/media/archivos/UMotorsport/video_yolo/'
        name = 'video_{:0>4d}.jpg'
        visualizer.save_in_video(path, name)
        connect_mng.close_connection()
