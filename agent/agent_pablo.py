from agent.agent import Agent
import numpy as np
from globals.globals import * # Global variables and constants, as if they were here
import math


class Agent_Pablo(Agent):
    '''
    Main Agent class, with basic features. All other agents inherit from this one, adding specific functions.

    '''

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
            agent_act['acc'] = 1.0
        else:
            agent_act['acc'] = 0.0

        # STEER CONTROL
        fpsM = sum(ifps) / len(ifps) if ifps else 0
        print(f'\nFPS: {fpsM}')
        angleDegrees = -1
        if (len(yellows) > 1 and len(blues) > 1):
            middleCone0 = [(blues[0]['coords']['x'] + yellows[0]['coords']['x']) / 2,
                           (blues[0]['coords']['y'] + yellows[0]['coords']['y']) / 2]
            middleCone1 = [(blues[1]['coords']['x'] + yellows[1]['coords']['x']) / 2,
                           (blues[1]['coords']['y'] + yellows[1]['coords']['y']) / 2]

            #extra = middleCone0[0] *(((car_state['speed'] + 0.01)/fpsM) + (agent_act['acc']/(2*(fpsM**2)))) if ifps else 0
            extra = (middleCone0[0] * fpsM) / (car_state['speed'] + 0.01) if fpsM else 0  # 8.5 = AVERAGE FPS
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
                agent_act['acc'] = 0.5
            # if (50>abs(angle)>0 and (blues[0]['coords']['x']<yellows[0]['coords']['x'] or blues[0]['coords']['y']<yellows[0]['coords']['y'])):
            agent_act['steer'] = angle / 25.0
        else:
            if maxSpeed > car_state['speed'] > (maxSpeed / 2):
                agent_act['acc'] = 0.2
            agent_act['steer'] = 1 if len(yellows) > len(blues) else lastAngle[0] / 60 if (
                    len(yellows) == 0 and len(yellows) == len(blues)) else 0 if len(yellows) == len(blues) else -1
            print(lastAngle)  # lastAngle[0]/75#