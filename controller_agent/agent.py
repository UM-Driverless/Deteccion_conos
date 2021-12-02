from controller_agent.agent_base import AgentInterface
from trayectory_estimation.cone_processing import ConeProcessing
import cv2
import simple_pid
import numpy as np


class Agent(AgentInterface):
    def __init__(self):
        self.initializeTrackbarsPID([4, 7, 15])
        self.pid = simple_pid.PID
        self.cone_processing = ConeProcessing()

    def initializeTrackbarsPID(self, intialTracbarVals):
        cv2.namedWindow("Trackbars PID")
        cv2.resizeWindow("Trackbars", 360, 240)
        cv2.createTrackbar("Kp/1000", "Trackbars", intialTracbarVals[0], 500, self.nothing)
        cv2.createTrackbar("Ki/1000", "Trackbars", intialTracbarVals[1], 500, self.nothing)
        cv2.createTrackbar("Kd/1000", "Trackbars", intialTracbarVals[2], 500, self.nothing)

    def valTrackbarsPID(self):
        kp = cv2.getTrackbarPos("Kp/1000", "Trackbars") / 1000
        ki = cv2.getTrackbarPos("Ki/1000", "Trackbars") / 1000
        kd = cv2.getTrackbarPos("Kd/1000", "Trackbars") / 1000

        return kp, ki, kd

    def nothing(self, x):
        pass

    def get_action(self, detections, segmentations):
        """ Calcular los valores de control
        """
        detections, labels = detections

        up_vertex = detections[0, :, 0, :]
        down_vertex = detections[0, :, 1, :]
        cone_bases = np.array([(up_vertex[:, 0] + (down_vertex[:, 0] - up_vertex[:, 0]) / 2.).astype(dtype=np.int), down_vertex[:, 1]])
        # cone_bases = detections
        # eagle_img = np.zeros(10, 10)
        # image_shape = (20, 20)
        # ref_point = self.create_cone_map(detections, cone_bases, eagle_img, image_shape)
        #
        # # Creo la imagen de coordenadas y pinto los conos
        # coord_img = np.zeros((image_shape[1], image_shape[1], image_shape[2]), dtype=np.uint8)
        #
        # img_center = int(coord_img.shape[0] / 2)
        #
        # turn_point = img_center - ref_point
        # val = self.valTrackbarsPID()
        # pid = self.pid(Kp=val[0], Ki=val[1], Kd=val[2], setpoint=0, output_limits=(-1., 1.))
        # steer = pid(turn_point)
        steer = 0.
        throttle = 0.05
        brake = 0.0
        clutch = 0.0
        upgear = 0
        downgear = 0

        return [throttle, brake, steer, clutch, upgear, downgear]


    def create_cone_map(self, detections, cone_centers, eagle_img, image_shape):
        return self.cone_processing.create_cone_map(detections, cone_centers, [eagle_img, image_shape])