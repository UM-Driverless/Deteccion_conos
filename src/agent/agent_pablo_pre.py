import math

from trajectory_estimation.cone_processing import ConeProcessing  # ConeProcessingNoWrapped
# import cv2
from simple_pid import PID
# import numpy as np
# from visualization_utils.logger import Logger
# from simple_pid import PID

class Agent:
    """
    Main Agent class, with basic features. All other agents inherit from this one, adding specific functions.
    """

    def __init__(self):
        self.cone_processing = ConeProcessing()
        self.speed_target = 5
        self.pid = PID(tauP, tauI, tauD, setpoint=0)

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

    def get_target(self, cones, car_state, actuation):
        """"
        Update actuation, calculated from the cones and car_state.
        """

        # SORT CONES FROM CLOSEST TO FURTHEST
        def take_x(cone):
            return cone['coords']['x']

        blues = [cone for cone in cones if (cone['label'] == 'blue_cone')]
        blues.sort(key=take_x)

        yellows = [cone for cone in cones if (cone['label'] == 'yellow_cone')]
        yellows.sort(key=take_x)

        fps = sum(ifps) / len(ifps) if ifps else 0

        if not fallo[0]:
            orange = [cone for cone in cones if (cone['label'] == 'orange_cone')]
            orange.sort(key=take_x)

            # SPEED CONTROL - actuation ----- Take (target speed - current speed) -> PID
            actuation['acc'] = (self.speed_target - car_state['speed']) * 0.1
            brake_condition = (len(orange) >= 6) and (orange[0]['coords']['y'] < 1)

            # If negative acceleration, brake instead
            if actuation['acc'] < 0:
                actuation['brake'] = -actuation['acc']
                actuation['acc'] = 0

            if (car_state['speed'] < maxSpeed) and (not brake_condition):
                actuation['acc'] = 1.0

            elif brake_condition:  # da igual la velocidad, si ve conos naranjas
                actuation['steer'] = 0  # 1 left, -1 right, 0 neutral
                actuation['acc'] = 0.0
                actuation['brake'] = 1.0

                if car_state['speed'] < 0.25:  # Si se ha parado completamente, AS_Finished
                    return True
            else:
                actuation['acc'] = 0.0

            # STEER CONTROL
            if (len(blues) > 0) and (len(yellows) > 0):

                go_y = (blues[0]['coords']['y'] + yellows[0]['coords']['y']) / 2  # positive means left
                go_x = (blues[0]['coords']['x']+ yellows[0]['coords']['x'])/2
                error = math.atan(go_y/go_x)*180/math.pi

                angle = min(max(self.pid(-error),-25),25) #configurar con angulo de giro maximo

                print(f'fps:           {fps:.4f}')
                print(f'speed:         {car_state["speed"]:.4f}\tmeters/second')
                print(f'distance:      {go_x:.4f}\tmeters')
                print(f'deviation:     {go_y:.4f}\tmeters')
                print(f'error:         {error:.4f}\tdegrees')
                print(f'steer angle:   {angle:.4f}\tdegrees')

                actuation['steer'] = angle/25  # /25, en simulador 1.0 corresponde a 25º

            elif len(blues) > 0:
                print("Bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
                actuation['steer'] = -1  # Rotation in Z axis. - = right

            elif len(yellows) > 0:
                print("Yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
                actuation['steer'] = 1  # Rotation in Z axis. + = left

            else:
                fallo[0] = True
                actuation['acc'] = 0.0
                actuation['brake']=1.0
                actuation['steer'] = 0.0

        else:
            actuation['acc'] = 0.0
            actuation['brake'] = 1.0
            actuation['steer'] = 0.0