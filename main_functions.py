from connection_utils.car_comunication import ConnectionManager
from controller_agent.agent import AgentAccelerationYoloFast as AgentAcceleration
from controller_agent.agent_ebs import AgentTest180 as AgentTest
from controller_agent.agent_ebs import AgentEBS as AgentEbs
from controller_agent.agent_ebs import AgentInspection as AgentInspection
from cone_detection.yolo_detector import ConeDetector
from visualization_utils.visualizer_yolo_det import Visualizer
from visualization_utils.logger import Logger
from globals import can_constants
import os
import time
import cv2

import pyzed.sl as sl
import numpy as np
import main_functions as mf

def inicio_de_movimiento(connect_mng):
    time.sleep(1)
    print('-----inicio mov 2.1-----')
    connect_mng.send_actions(throttle=0.3,
                             brake=0,
                             steer=0,
                             clutch=2,
                             upgear=0,
                             downgear=0)
    time.sleep(2)
    print('-----inicio mov 2.2-----')
    connect_mng.send_actions(throttle=0.3,
                             brake=0,
                             steer=0,
                             clutch=1,
                             upgear=0,
                             downgear=0)

    print('-----inicio mov 2.3-----')
    connect_mng.send_actions(throttle=0.3,
                             brake=0,
                             steer=0,
                             clutch=0,
                             upgear=0,
                             downgear=0)

def arranque(connect_mng):
    # connect_mng.do_read_msg()
    check = 0
    # comprobar que el motor esta encendido
    while check <= 3:
        connect_mng.send_actions(throttle=0,
                                 brake=0,
                                 steer=0,
                                 clutch=2,
                                 upgear=0,
                                 downgear=0)
        rpm = connect_mng.can.get_rpm_can()
        print(f"----RPM: {rpm}")
        rpm = 2000
        if rpm < can_constants.RPM_RALENTI:
            time.sleep(0.5)
        else:
            check += 1
            print('-----CHECK----')
    print('-----coche arrancado-----')
    time.sleep(10)

def seleccion_agente_arrancado(connect_mng, agente, logger):
    # Inicializar Agente (controlador)
    if agente == 1:
        print('---------------ACCELERATION--------------')
        agent_init = AgentAcceleration(logger=logger, target_speed=20.)
        arranque(connect_mng)
    elif agente == 2:
        print('---------------Skidpad--------------')
        #agent_init = AgentTest(logger=logger, target_speed=5.)
        return 0
    elif agente == 3:
        print('---------------Autocross--------------')
        #agent_init = AgentTest(logger=logger, target_speed=5.)
        return 0
    elif agente == 4:
        print('---------------Trackdrive--------------')
        #agent_init = AgentTest(logger=logger, target_speed=5.)
        return 0
    elif agente == 5:
        print('---------------EBS test--------------')
        agent_init = AgentEbs(logger=logger, target_speed=40.)
        arranque(connect_mng)
    elif agente == 6:
        print('---------------Inspection--------------')
        agent_init = AgentInspection(logger=logger)
        arranque(connect_mng)
    elif agente == 7:
        print('---------------Manual driving--------------')
        # agent_init = AgentTest(logger=logger, target_speed=5.)
        return 0
    elif agente == 8:
        print('---------------TEST180--------------')
        agent_init = AgentTest(logger=logger, target_speed=5.)
        arranque(connect_mng)
    print('agent initialized')

    return agent_init