from trajectory_estimation.cone_processing import ConeProcessing #ConeProcessingNoWrapped
import cv2
import simple_pid
import numpy as np
from visualization_utils.logger import Logger

from globals.globals import * # Global variables and constants, as if they were here

# TODO SIMPLE AGENT THAT INHERITS FROM GENERAL

class Agent():
    '''
    Main Agent class, with all the common actions between agents.
    Each different test will inherit from this class and implement the specific actions for the test.
    '''
    def __init__(self, logger, target_speed=20.):
        # TODO TEST Agent Variables
        ## PID
        self.pid_acc_kp = 1
        self.pid_acc_ki = 1
        self.pid_acc_kd = 1
        
        
        self.pid_kp = 4
        self.pid_ki = 7
        self.pid_kd = 15
        self.pid_throttle_kp = 700
        self.pid_throttle_ki = 300
        self.pid_throttle_kd = 500
        self.pid_brake_kp = 10
        self.pid_brake_ki = 25
        self.pid_brake_kd = 40


        # PID
        self.initializeTrackbarsPID()
        self.pid_steer = simple_pid.PID
        self.pid_throttle = simple_pid.PID
        self.pid_brake = simple_pid.PID
        
        # TODO test
        self.pid_acc = simple_pid.PID


        self.cone_processing = ConeProcessing()

        # Controla a que velocidad se suelta el embrague por completo
        self.clutch_max_speed = 10.
        self.clutch_max_rpm = 2000.

        self.target_speed = target_speed

    def initializeTrackbarsPID(self):
        '''
        Creates Trackbar windows with default values
        '''
        
        cv2.namedWindow("Trackbars PID")
        cv2.resizeWindow("Trackbars PID", 360, 240)
        cv2.createTrackbar("Kp/1000", "Trackbars PID", 4, 500, self.nothing)
        cv2.createTrackbar("Ki/1000", "Trackbars PID", 7, 500, self.nothing)
        cv2.createTrackbar("Kd/1000", "Trackbars PID", 15, 500, self.nothing)
        cv2.createTrackbar("Throttle Kp/100", "Trackbars PID", 700, 1000, self.nothing)
        cv2.createTrackbar("Throttle Ki/100", "Trackbars PID", 300, 1000, self.nothing)
        cv2.createTrackbar("Throttle Kd/100", "Trackbars PID", 500, 1000, self.nothing)
        cv2.createTrackbar("Brake Kp/100", "Trackbars PID", 10, 1000, self.nothing)
        cv2.createTrackbar("Brake Ki/1000", "Trackbars PID", 25, 1000, self.nothing)
        cv2.createTrackbar("Brake Kd/1000", "Trackbars PID", 40, 1000, self.nothing)
        
        # TODO test new trackbar
        cv2.namedWindow("PID settings (Acceleration, and steering)")
        cv2.resizeWindow("Trackbars PID", 360, 240)
        cv2.createTrackbar("Accel","PID settings (Acceleration, and steering)",0,100,self.nothing)
        

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

    def nothing(self, x):
        pass

    def get_action(self, detections, segmentations, speed, rpm, orig_im_shape=(1, 180, 320, 3)):
        """
        Calcular los valores de control
        """
        centers, detections, labels = detections

        eagle_img = None
        image_shape = None
        data = self.cone_processing.create_cone_map(centers, labels, [eagle_img, image_shape])

        img_center = int(orig_im_shape[2] / 2)
        steer = self.horizontal_control(ref_point=data[-1], img_center=img_center)

        throttle, brake, clutch = self.longitudinal_control(cenital_cones=data[1], speed=speed, rpm=rpm)

        return [throttle, brake, steer, clutch], data

    def horizontal_control(self, ref_point, img_center):
        turn_point = img_center - ref_point
        val = self.valTrackbarsPID()
        pid = self.pid_steer(Kp=val[0], Ki=val[1], Kd=val[2], setpoint=0, output_limits=(-1., 1.))

        return pid(turn_point)

    def longitudinal_control(self, cenital_cones, speed, rpm):
        blue_center, yell_center, oran_left_center, oran_rigth_center = cenital_cones

        n_color_cones = len(blue_center) + len(yell_center)
        n_oran_cones = len(oran_left_center) + len(oran_rigth_center)

        # if n_color_cones > n_oran_cones:
        #     throttle = 1.0
        #     brake = 0.0
        # else:
        #     throttle = 0.0
        #     brake = 1.0

        val = self.valTrackbarsPID()
        pid_throttle = self.pid_throttle(Kp=val[3], Ki=val[4], Kd=val[5], setpoint=0, output_limits=(0., 1.))
        pid_brake = self.pid_brake(Kp=val[6], Ki=val[7], Kd=val[8], setpoint=0, output_limits=(0., 1.))

        if n_color_cones > n_oran_cones:
            target_speed = self.target_speed

            ref_point = speed - target_speed
            throttle = pid_throttle(ref_point)
            brake = 0.
        else:
            throttle = 0.
            ref_point = - speed
            brake = pid_brake(ref_point)

        print(throttle, brake, ref_point, val[3], val[4], val[5], val[6], val[7], val[8])

        agent_target = self.clutch_func(car_state, agent_target)

        # upgear, downgear, gear = self.change_gear(gear, rpm, throttle)

        return agent_target

class AgentYolo(Agent):
    '''
    Based on Agent, adds some specific changes
    
    TODO what changes?
    '''
    
    def __init__(self, logger, target_speed=20.):
        super().__init__(logger=logger)

    def get_action_sim(self, agent_target, cones, image, sim_client2, simulator_car_controls):
        """
        Figure out what to do to drive the car. Updates agent_target values.
        TODO rename agent_target so it's different than the global name?
        
        """
        # SPEED
        sim_state = sim_client2.getCarState()
        car_state['speed'] = sim_state.speed
        
        
        if (sim_state.speed < 5):
            agent_target['acc'] = 0.5
        else:
            agent_target['acc'] = 0.0
        
        # STEER
        def sort_func(cone): return cone['coords']['x']
        blues = [cone for cone in cones if (cone['label'] == 'blue_cone')]
        blues.sort(key=sort_func)
        yellows = [cone for cone in cones if (cone['label'] == 'yellow_cone')]
        yellows.sort(key=sort_func)
        
        if (len(blues) > 0) and (len(yellows) > 0):
            # I assume they're sorted from closer to further
            center = (blues[0]['coords']['y'] + yellows[0]['coords']['y']) / 2
            print(f'center:{center}')
            agent_target['steer'] = center * 0.5 # -1 left, 1 right, 0 neutral TODO HACER CON MAS SENTIDO
        elif len(blues) > 0:
            agent_target['steer'] = 1 # -1 left, 1 right, 0 neutral
        elif len(yellows) > 0:
            agent_target['steer'] = -1 # -1 left, 1 right, 0 neutral
        
        simulator_car_controls.steering = agent_target['steer']
        simulator_car_controls.throttle = agent_target['acc']
        sim_client2.setCarControls(simulator_car_controls)

        return agent_target

    def get_action(self, agent_target, cones, image):
        
        "Actual agent without the sim"

        return agent_target

    def horizontal_control2(self, agent_target, cones, client, simulator_car_controls):
        agent_target['steer'] = 0.5 # -1 left, 1 right, 0 neutral
        simulator_car_controls.steering = agent_target['steer']
        client.setCarControls(simulator_car_controls)

        return agent_target
    
    def longitudinal_control2(self, agent_target, cones, client, simulator_car_controls):
        agent_target['acc'] = 0.1
        # TODO ACHIEVE TARGET SPEED
        simulator_car_controls.throttle = agent_target['acc']
        client.setCarControls(simulator_car_controls)
                
        return agent_target
    
    def horizontal_control(self, target=0, img_center=0):
        '''
        TODO MAKE IT WORK
        '''
        turn_point = img_center - target
        # TODO HERE PRINT WITH LOGGER TURN POINT
        # turn_point = -turn_point/img_base_len
        val = self.valTrackbarsPID()
        pid = self.pid_steer(Kp=val[0], Ki=val[1], Kd=val[2], setpoint=0, output_limits=(-1., 1.))
        return pid(turn_point)

    def longitudinal_control(self, agent_target, car_state, cones):
        
        blue_center, yell_center, oran_left_center, oran_rigth_center, _, _ = [
            [cone['coords'] for cone in cones if cone['label'] == 'blue_cone'],
            [cone['coords'] for cone in cones if cone['label'] == 'yellow_cone'],
            [cone['coords'] for cone in cones if cone['label'] == 'orange_cone'],
            [cone['coords'] for cone in cones if cone['label'] == 'large_orange_cone'],
            [cone['coords'] for cone in cones if cone['label'] == 'unknown_cone'],
            [cone['coords'] for cone in cones if cone['label'] == 'unknown_cone']
        ]

        # Calculate the amount of blue or yellow cones, which means run, and orange cones, which means stop.
        n_color_cones = len(blue_center) + len(yell_center)
        n_oran_cones = len(oran_left_center) + len(oran_rigth_center)

        # Get PID parameters from the trackbars
        val = self.valTrackbarsPID() # TODO update self values and let the code use them instead of passing between functions. Same class.
        
        
        pid_throttle = self.pid_throttle(Kp=val[3], Ki=val[4], Kd=val[5], setpoint=0, output_limits=(0., 1.))
        pid_brake = self.pid_brake(Kp=val[6], Ki=val[7], Kd=val[8], setpoint=0, output_limits=(0., 1.))

        if n_color_cones > n_oran_cones:
            # Not the end of the circuit
            target_speed = self.target_speed

            ref_point = car_state['speed'] - target_speed
            agent_target['throttle'] = pid_throttle(ref_point)
            # throttle = 1.0

            agent_target['brake'] = 0.
        else:
            # More orange than color. Braking zone
            blue_braking_zone = False
            for i in range(oran_left_center.shape[0]):
                for j in range(blue_center.shape[0]):
                    if oran_left_center[i, 1] > blue_center[j, 1]:
                        blue_braking_zone = True

            yell_braking_zone = False
            for i in range(oran_rigth_center.shape[0]):
                for j in range(yell_center.shape[0]):
                    if oran_rigth_center[i, 1] > yell_center[j, 1]:
                        yell_braking_zone = True

            if blue_center.shape[0] < 1:
                blue_braking_zone = True
            if yell_center.shape[0] < 1:
                yell_braking_zone = True

            if not blue_braking_zone and not yell_braking_zone:  # No braking zone
                target_speed = self.target_speed

                ref_point = car_state['speed'] - target_speed
                agent_target['throttle'] = pid_throttle(ref_point)
                # throttle = 1.0
                agent_target['brake'] = 0.
            else: # Braking zone
                agent_target['throttle'] = 0.
                ref_point = - car_state['speed']
                agent_target['brake'] = pid_brake(ref_point)

        #print(throttle, brake, ref_point, val[3], val[4], val[5], val[6], val[7], val[8])

        # upgear, downgear, gear = self.change_gear(gear, rpm, throttle)
        return agent_target