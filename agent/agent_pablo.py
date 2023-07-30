from trajectory_estimation.cone_processing import ConeProcessing  # ConeProcessingNoWrapped
# import cv2
# import simple_pid
import numpy as np
import math
# from visualization_utils.logger import Logger
# from simple_pid import PID

from globals.globals import *  # Global variables and constants, as if they were here


class Agent():
    '''
    Main Agent class, with basic features. All other agents inherit from this one, adding specific functions.

    '''

    def __init__(self):
        self.cone_processing = ConeProcessing()

    """
    def valTrackbarsPID(self):
        '''
        returns the information from the trackbars
        '''
        kp = cv2.getTrackbarPos("Kp/1000", "Trackbars PID") / 1000
        ki = cv2.getTrackbarPos("Ki/1000", "Trackbars PID") / 1000
        kd = cv2.getTrackbarPos("Kd/1000", "Trackbars PID") / 1000
        throttle_kp = cv2.getTrackbarPos("Throttle Kp/100", "Trackbars PID") / 1000
        throttle_ki = cv2.getTrackbarPos("Throttle Ki/100", "Trackbars PID") / 1000
        throttle_kd = cv2.getTrackbarPos("Throttle Kd/100", "Trackbars PID") / 1000
        brake_kp = cv2.getTrackbarPos("Brake Kp/100", "Trackbars PID") / 1000
        brake_ki = cv2.getTrackbarPos("Brake Ki/1000", "Trackbars PID") / 1000
        brake_kd = cv2.getTrackbarPos("Brake Kd/1000", "Trackbars PID") / 1000

        # TODO TEST update self values instead of return
        self.pid_kp = kp
        self.pid_ki = ki
        self.pid_kd = kd
        self.pid_throttle_kp = throttle_kp
        self.pid_throttle_ki = throttle_ki
        self.pid_throttle_kd = throttle_kd
        self.pid_brake_kp = brake_kp
        self.pid_brake_ki = brake_ki
        self.pid_brake_kd = brake_kd

        return kp, ki, kd, throttle_kp, throttle_ki, throttle_kd, brake_kp, brake_ki, brake_kd
    """

    def get_target(self, cones, car_state, agent_act):
        '''
        Update agent_act, calculated from the cones and car_state.
        '''

        # SORT CONES FROM CLOSEST TO FURTHEST
        def take_x(cone):
            return cone['coords']['x']

        blues = [cone for cone in cones if (cone['label'] == 'blue_cone')]
        blues.sort(key=take_x)

        yellows = [cone for cone in cones if (cone['label'] == 'yellow_cone')]
        yellows.sort(key=take_x)

        # SPEED CONTROL - agent_act ----- Take (target speed - current speed) -> PID
        if (car_state['speed'] < maxSpeed):
            agent_act['acc_normalized'] = 1.0
        else:
            agent_act['acc_normalized'] = 0.0

        # STEER CONTROL

        print(f'\n{sum(ifps) / (len(ifps)) if ifps else 0}')
        angleDegrees = -1
        if (len(yellows) > 1 and len(blues) > 1):
            middleCone0 = [(blues[0]['coords']['x'] + yellows[0]['coords']['x']) / 2,
                           (blues[0]['coords']['y'] + yellows[0]['coords']['y']) / 2]
            middleCone1 = [(blues[1]['coords']['x'] + yellows[1]['coords']['x']) / 2,
                           (blues[1]['coords']['y'] + yellows[1]['coords']['y']) / 2]
            extra = (middleCone0[0] * (sum(ifps) / (len(ifps)))) / (
                        car_state['speed'] + 0.01) if ifps else 0  # 8.5 = AVERAGE FPS
            A = np.array(
                [[0, 0, 1], [middleCone0[0] ** 2, middleCone0[0], 1], [middleCone1[0] ** 2, middleCone1[0], 1]])
            B = np.array([0, middleCone0[1], middleCone1[1]])
            X = np.linalg.solve(A, B)
            movetoP = X[0] * (extra ** 2) + X[1] * extra + X[2]
            movetoD = 2 * X[0] * extra + X[1]
            movetoI = X[0] * ((extra ** 3) / 3) + X[1] * ((extra ** 2) / 2) + X[2] * extra
            lY1 = 0  # line[0][1]
            lX1 = 0  # line[0][0]
            line1Y2 = 1  # lineA[1][1]
            line1X2 = 0  # lineA[1][0]
            line2Y2p = movetoP  # lineB[1][1]
            line2Y2d = movetoD  # lineB[1][1]
            line2Y2i = movetoI  # lineB[1][1]
            line2X2 = extra  # lineB[1][0]
            # calculate angle between pairs of lines
            angle1 = math.atan2(lY1 - line1Y2, lX1 - line1X2)
            angle2p = math.atan2(lY1 - line2Y2p, lX1 - line2X2)
            angleDegreesp = (angle1 - angle2p) * 360 / (2 * math.pi)
            angle2d = math.atan2(lY1 - line2Y2d, lX1 - line2X2)
            angleDegreesd = (angle1 - angle2d) * 360 / (2 * math.pi)
            angle2i = math.atan2(lY1 - line2Y2i, lX1 - line2X2)
            angleDegreesi = (angle1 - angle2i) * 360 / (2 * math.pi)
            print(f'p: {angleDegreesp}\td: {angleDegreesd}\ti: {angleDegreesi}')
            if 90 > angleDegreesp > 30:
                angleDegreesp = 90 - angleDegreesp
            elif -210 > angleDegreesp > -270:
                angleDegreesp = -270 - angleDegreesp
            else:
                angleDegreesp = 0
            if 90 > angleDegreesd > 30:
                angleDegreesd = 90 - angleDegreesd
            elif -210 > angleDegreesd > -270:
                angleDegreesd = -270 - angleDegreesd
            else:
                angleDegreesd = 0
            if 90 > angleDegreesi > 30:
                angleDegreesi = 90 - angleDegreesi
            elif -210 > angleDegreesi > -270:
                angleDegreesi = -270 - angleDegreesi
            else:
                angleDegreesi = 0
            angle = angleDegreesp * tauP + angleDegreesi * tauI + angleDegreesd * tauD
            print(f'p: {angleDegreesp}\td: {angleDegreesd}\ti: {angleDegreesi}')
            print(f'{angle}\t{angle / 25}')
            lastAngle[0] = angle

            if (0 < angleDegrees < 70 or -270 < angleDegrees < -250) and maxSpeed > car_state['speed'] > (maxSpeed / 2):
                agent_act['acc_normalized'] = 0.5
            # if (50>abs(angle)>0 and (blues[0]['coords']['x']<yellows[0]['coords']['x'] or blues[0]['coords']['y']<yellows[0]['coords']['y'])):
            agent_act['steer'] = angle / 25.0
        else:
            if maxSpeed > car_state['speed'] > (maxSpeed / 2):
                agent_act['acc_normalized'] = 0.2
            agent_act['steer'] = 1 if len(yellows) > len(blues) else lastAngle[0] / 60 if (
                    len(yellows) == 0 and len(yellows) == len(blues)) else 0 if len(yellows) == len(blues) else -1
            print(lastAngle)  # lastAngle[0]/75#
        """
        if (len(blues) > 0) and (len(yellows) > 0):
            # I assume they're sorted from closer to further
            center = (blues[0]['coords']['y'] + yellows[0]['coords']['y']) / 2 # positive means left
            # print(f'center:{center}')
            agent_act['steer'] = center * 0.5 # -1 left, 1 right, 0 neutral TODO HACER CON MAS SENTIDO
        elif len(blues) > 0:
            agent_act['steer'] = -1 # Rotation in Z axis. - = right
        elif len(yellows) > 0:
            agent_act['steer'] = +1 # Rotation in Z axis. + = left
        else:
            agent_act['steer'] = 1.0 # Rotation in Z axis. + = left"""