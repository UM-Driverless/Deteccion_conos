from agent.agent import Agent
import numpy as np
import math


class Agent_Pablo(Agent):
    '''
    Main Agent class, with basic features. All other agents inherit from this one, adding specific functions.

    '''

    def puntos_en_parabola(self,a, b, k):
        # Resolver la ecuación cuadrática para obtener los puntos en la parábola
        x = np.linspace(-100, 100, 1000)  # Rango de valores para x
        y = a * x ** 2 + b * x

        # Calcular la distancia de cada punto a (0, 0)
        distancia = np.sqrt(x ** 2 + y ** 2)

        # Encontrar los índices de los puntos que están a una distancia k de (0, 0)
        indices = np.where(np.isclose(distancia, k))

        # Obtener los valores de x correspondientes a esos índices
        puntos_x = x[indices]

        return puntos_x
    def plotParabola(self,X):
        import matplotlib.pyplot as plt
        # Generate x-values for plotting
        x = np.linspace(-10, 10, 400)  # Adjust the range as needed
        # Calculate corresponding y-values using the quadratic equation
        y = X[0] * x ** 2 + X[1] * x
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label=f'{X[0]}x^2 + {X[1]}x')
        plt.title('Plot of a Parabola')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.grid()
        plt.legend()
        plt.show()
        plt.savefig('/home/hydra/Documents/UM/image.png')


    def get_target(self, cones, car_state, actuation):
        '''
        Update actuation, calculated from the cones and car_state.
        '''

        # SORT CONES FROM CLOSEST TO FURTHEST
        def take_x(cone):
            return cone['coords']['x']

        blues = [cone for cone in cones if (cone['label'] == 'blue_cone')]
        blues.sort(key=lambda c:((c['coords']['x']**2+c['coords']['y']**2)**1/2))

        yellows = [cone for cone in cones if (cone['label'] == 'yellow_cone')]
        yellows.sort(key=lambda c:((c['coords']['x']**2+c['coords']['y']**2)**1/2))

        # SPEED CONTROL - actuation ----- Take (target speed - current speed) -> PID
        if car_state['speed'] < maxSpeed:
            actuation['acc'] = 1.0
        else:
            actuation['acc'] = 0.0

        # STEER CONTROL
        fpsM = sum(ifps) / len(ifps) if ifps else 0
        print(f'\nFPS: {fpsM}')
        if (len(yellows) > 2 and len(blues) > 2):
            middleCone0 = [(blues[0]['coords']['x'] + yellows[0]['coords']['x']) / 2,
                           (blues[0]['coords']['y'] + yellows[0]['coords']['y']) / 2]
            middleCone1 = [(blues[1]['coords']['x'] + yellows[1]['coords']['x']) / 2,
                           (blues[1]['coords']['y'] + yellows[1]['coords']['y']) / 2]
            middleCone2 = [(blues[2]['coords']['x'] + yellows[2]['coords']['x']) / 2,
                           (blues[2]['coords']['y'] + yellows[2]['coords']['y']) / 2]
        #if (len(yellows) > 1 and len(blues) > 1 and ((middleCone0[0]>0 and middleCone1[0]>0)or(middleCone0[0]<0 and middleCone1[0]<0))):
            print(middleCone0)
            print(middleCone1)
            k = (car_state['speed']/(fpsM))  if ifps else 0 # + (actuation['acc']/(2*(fpsM**2))) #modulo recta a intersectar
            extra = 0
            A = np.array([[0, 0, 1], [middleCone0[0] ** 2, middleCone0[0], 1], [middleCone1[0] ** 2, middleCone1[0], 1]])
            B = np.array([0, middleCone0[1], middleCone1[1]])
            X = np.linalg.solve(A, B)

            """x1 = (-X[1])/(2*X[0])-k
            x2 = (-X[1])/(2*X[0])+k"""

            """if middleCone0[0] > 0:
                if x1>0 and x2>0:
                    if x1<x2:
                        extra=x1
                    else:
                        extra=x2
                elif x1>0:
                    extra=x1
                elif x2>0:
                    extra=x2
            else:
                if x1 < 0 and x2 < 0:
                    if x1 > x2:
                        extra = x1
                    else:
                        extra = x2
                elif x1 < 0:
                    extra = x1
                elif x2 < 0:
                    extra = x2"""
            """if middleCone0[0] > 0:
                if x1 > x2:
                    extra = x1
                else:
                    extra = x2
            else:
                if x1 < x2:
                    extra = x1
                else:
                    extra = x2"""
            sols = self.puntos_en_parabola(X[0],X[1],k)
            print(sols)
            extra = sorted(sols,key=lambda s:(X[0]*(s**2)+X[1]*s))[0] if sols else 0




            print(f'parabola: {X}')
            print(f'k:{k}  extra:{extra}  sols:{sols}')


            self.plotParabola(X)


            movetoP = X[0] * (extra ** 2) + X[1] * extra
            movetoD = 2 * X[0] * extra + X[1]
            movetoI = X[0] * ((extra ** 3) / 3) + X[1] * ((extra ** 2) / 2)
            lY1 = 0  # line[0][1]
            lX1 = 0  # line[0][0]
            line1Y2 = 1  # lineA[1][1]
            line1X2 = 0  # lineA[1][0]
            line2Y2p = movetoP  # lineB[1][1]
            line2Y2d = movetoD  # lineB[1][1]
            line2Y2i = movetoI  # lineB[1][1]
            line2X2 = extra  # lineB[1][0]
            # calculate angle between pairs of lines
            PID=[0,0,0]
            angle1 = math.atan2(lY1 - line1Y2, lX1 - line1X2)
            angle2p = math.atan2(lY1 - line2Y2p, lX1 - line2X2)
            PID[0] = (angle1 - angle2p) * 360 / (2 * math.pi)
            angle2d = math.atan2(lY1 - line2Y2d, lX1 - line2X2)
            PID[2] = (angle1 - angle2d) * 360 / (2 * math.pi)
            angle2i = math.atan2(lY1 - line2Y2i, lX1 - line2X2)
            PID[1] = (angle1 - angle2i) * 360 / (2 * math.pi)

            print(f'p: {PID[0]}\td: {PID[2]}\ti: {PID[1]}')
            """
            for i in range(3):
                if 90 > PID[i] > 20:
                    PID[i] = 90 - PID[i]
                elif -200 > PID[i] > -270:
                    PID[i] = -270 - PID[i]
                else:
                    PID[i] = 0
            """
            for i in range(3):
                if PID[i]<0:
                    PID[i]=-(PID[i]%90)
                else:
                    PID[i]=(PID[i]%90)








            """
            angleDegreesp = math.atan2(extra, movetoP) * 360 / (2 * math.pi)
            angleDegreesd = math.atan2(extra, movetoI) * 360 / (2 * math.pi)
            angleDegreesi = math.atan2(extra, movetoD) * 360 / (2 * math.pi)
            """
            angle = PID[0] * tauP #+ PID[1] * tauI + PID[2] * tauD
            print(f'p: {PID[0]}\td: {PID[2]}\ti: {PID[1]}')
            print(f'{angle}\t{angle / 25}')
            lastAngle[0] = angle

            if (abs(angle)>30) and maxSpeed > car_state['speed'] > (maxSpeed / 2):
                actuation['acc'] = 0.5
            # if (50>abs(angle)>0 and (blues[0]['coords']['x']<yellows[0]['coords']['x'] or blues[0]['coords']['y']<yellows[0]['coords']['y'])):
            actuation['steer'] = angle / 25.0
        else:
            if maxSpeed > car_state['speed'] > (maxSpeed / 2):
                actuation['acc'] = 0.2
            actuation['steer'] = 1 if len(yellows) > len(blues) else lastAngle[0] / 60 if (
                    len(yellows) == 0 and len(yellows) == len(blues)) else 0 if len(yellows) == len(blues) else -1
            print(lastAngle)  # lastAngle[0]/75#


"""
ifps = []
maxSpeed = 7
tauP = 1.0#0.4#    0.6*Ku    0.2    0.09
tauD = 0.2#2.4#    2*Kp/Pu    3.0     1.40625
tauI = 0.1#0.009#    Kp*Pu/8    0.004   0.00144
lastAngle = [0]

prev_integral=[0,0]
fallo=[False]
"""