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

        # Controla a que velocidad se suelta el embrague por completo
        self.clutch_max_speed = 10.

    def initializeTrackbarsPID(self, intialTracbarVals):
        cv2.namedWindow("Trackbars PID")
        cv2.resizeWindow("Trackbars PID", 360, 240)
        cv2.createTrackbar("Kp/1000", "Trackbars PID", intialTracbarVals[0], 500, self.nothing)
        cv2.createTrackbar("Ki/1000", "Trackbars PID", intialTracbarVals[1], 500, self.nothing)
        cv2.createTrackbar("Kd/1000", "Trackbars PID", intialTracbarVals[2], 500, self.nothing)

    def valTrackbarsPID(self):
        kp = cv2.getTrackbarPos("Kp/1000", "Trackbars PID") / 1000
        ki = cv2.getTrackbarPos("Ki/1000", "Trackbars PID") / 1000
        kd = cv2.getTrackbarPos("Kd/1000", "Trackbars PID") / 1000

        return kp, ki, kd

    def nothing(self, x):
        pass

    def get_action(self, detections, segmentations, speed, gear, rpm, orig_im_shape=(1, 180, 320, 3)):
        """
        Calcular los valores de control
        """
        centers, detections, labels = detections

        data = self.create_cone_map(centers, labels, None, None)

        img_center = int(orig_im_shape[2] / 2)
        steer = self.horinzontal_control(ref_point=data[-1], img_center=img_center)

        throttle, brake, clutch, upgear, downgear = self.longitudinal_control(cenital_cones=data[1], speed=speed,  gear=gear, rpm=rpm)

        return [throttle, brake, steer, clutch, upgear, downgear], data


    def create_cone_map(self, centers, labels, eagle_img, image_shape):
        return self.cone_processing.create_cone_map(centers, labels, [eagle_img, image_shape])

    def horinzontal_control(self, ref_point, img_center):
        turn_point = img_center - ref_point
        val = self.valTrackbarsPID()
        pid = self.pid(Kp=val[0], Ki=val[1], Kd=val[2], setpoint=0, output_limits=(-1., 1.))

        return pid(turn_point)

    def longitudinal_control(self, cenital_cones, speed,  gear, rpm):
        blue_center, yell_center, oran_left_center, oran_rigth_center = cenital_cones

        n_color_cones = len(blue_center) + len(yell_center)
        n_oran_cones = len(oran_left_center) + len(oran_rigth_center)

        if n_color_cones > n_oran_cones:
            throttle = 1.0
            brake = 0.0
        else:
            throttle = 0.0
            brake = 1.0
        clutch = self.clutch_func(speed, throttle, brake)

        upgear, downgear = self.change_gear(gear, rpm, throttle)

        return throttle, brake, clutch, upgear, downgear

    def change_gear(self,  gear, rpm, throttle):
        upgear = 0.
        downgear = 0.

        if rpm > 4000 and gear >= 0 and gear < 4:
            upgear = 1.
        elif rpm < 1000. and gear > 0 and gear <= 4 and throttle <= 0.1:
            downgear = 1.
        return upgear, downgear

    def clutch_func(self, speed, throttle, brake):
        clutch = 0.
        if speed < self.clutch_max_speed:
            speed = speed/self.clutch_max_speed
            if speed < 0.1:
                if throttle > 0.25:
                    clutch = 0.8
                elif brake > 0.1:
                    clutch = 1.0
                else:
                    clutch = 1.0
            else:
                clutch = (0.2 / speed) - 0.2
        return clutch
