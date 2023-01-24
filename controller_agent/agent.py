from controller_agent.agent_base import AgentInterface
from trajectory_estimation.cone_processing import ConeProcessing #ConeProcessingNoWrapped
import cv2
import simple_pid
import numpy as np

from globals import * # Global variables and constants, as if they were here

class AgentAcceleration(AgentInterface):
    def __init__(self, logger, target_speed=20.):
        super().__init__(logger=logger)

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
        steer = self.horinzontal_control(ref_point=data[-2], img_center=img_center)

        throttle, brake, clutch = self.longitudinal_control(cenital_cones=data[1], speed=speed, rpm=rpm)

        return [throttle, brake, steer, clutch], data        

    def create_cone_map(self, centers, labels, eagle_img, image_shape):
        return self.cone_processing.create_cone_map(centers, labels, [eagle_img, image_shape])

    def horinzontal_control(self, ref_point, img_center):
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

    # def change_gear(self,  gear, rpm, throttle):
    #     upgear = 0.
    #     downgear = 0.

    #     if rpm > self.upgear_rpm and gear >= self.min_gear and gear < self.max_gear:
    #         upgear = 1.
    #         if self.gear < self.max_gear:
    #             self.gear +=1
    #     elif rpm < self.downgear_rpm and gear > self.min_gear and gear <= self.max_gear and throttle <= 0.1:
    #         downgear = 1.
    #         if self.gear > self.min_gear:
    #             self.gear -= 1
    #     return upgear, downgear, self.gear

    def clutch_func(self, car_state, agent_target):
        speed = car_state['speed']
        rpm = car_state['rpm']
        
        if speed < self.clutch_max_speed and rpm < self.clutch_max_rpm:
            speed = speed/self.clutch_max_speed
            if speed < 0.1:
                if agent_target['throttle'] > 0.25:
                    agent_target['clutch'] = 0.8
                elif agent_target['brake'] > 0.1:
                    agent_target['clutch'] = 1.0
                else:
                    agent_target['clutch'] = 1.0
            else:
                rpm = rpm / self.clutch_max_rpm
                agent_target['clutch'] = (0.2 / rpm) - 0.2

        return agent_target

'''#OLD
class AgentTestClutchThrottle(AgentAcceleration):
    def __init__(self, logger, target_speed=10.):
        super().__init__(logger=logger, target_speed=target_speed)
        # Controla a que velocidad se suelta el embrague por completo
        self.clutch_max_speed = 10.
        self.clutch_max_rpm = 2000.

        self.iter_arranque = 0.
        self.max_iter_arranque = 10.
        self.clutch_max_rpm_arranque = 3000.


    def get_action(self, detections, segmentations, speed, rpm, orig_im_shape=(1, 180, 320, 3)):
        """
        Calcular los valores de control
        """
        steer = 0.

        throttle, brake, clutch = self.longitudinal_control(None, speed=speed, rpm=rpm)

        return [throttle, brake, steer, clutch], None


    def longitudinal_control(self, cenital_cones, speed,  rpm):
        val = self.valTrackbarsPID()


        if self.iter_arranque  > self.max_iter_arranque:
            brake = 0.9
        else:
            brake = 0.
        throttle = self.throttle_func(speed, brake, rpm)

        clutch = self.clutch_func(car_state, agent_target)

        # upgear, downgear, gear = self.change_gear(gear, rpm, throttle)

        return throttle, brake, clutch

    def throttle_func(self, speed, brake, rpm):
        if rpm < self.clutch_max_rpm_arranque:
            if brake < 0.1:
                rpm = (rpm / self.clutch_max_rpm_arranque) * 10 - 10
                throttle = -1 / (rpm) - 0.1
            else:
                throttle = 0.
        else:
            throttle = 0.
        return throttle

    def clutch_func(self, car_state, agent_target):
        speed = car_state['speed']
        rpm = car_state['rpm']
        
        if speed < self.clutch_max_speed and rpm < self.clutch_max_rpm_arranque:
            speed = speed/self.clutch_max_speed
            if speed < 0.1:
                if agent_target['throttle'] > 0.25:
                    agent_target['clutch'] = 0.7
                elif agent_target['brake'] > 0.1:
                    agent_target['clutch'] = 0.9
                else:
                    agent_target['clutch'] = 0.9
            else:
                rpm = rpm / self.clutch_max_rpm_arranque
                agent_target['clutch'] = np.clip((0.2 / rpm) - 0.2, 0., 0.9)

        return agent_target
'''

class AgentAccelerationYolo(AgentAcceleration):
    def __init__(self, logger, target_speed=20.):
        super().__init__(logger=logger)

    def get_action(self, agent_target, car_state, detections, cone_centers=None, orig_im_shape=(1, 180, 320, 3), image=None):
        """
        Figure out what to do to drive the car. Updates agent_target values.
        
        """
        bboxes, labels = detections

        img_center = int(image.shape[1] / 2) # Usually (640,640,3) -> 320
        
        data = self.cone_processing.create_cone_map(cone_centers, labels, None, orig_im_shape=(1,) + image.shape, img_to_wrap=image)
        
        agent_target = self.longitudinal_control(agent_target, car_state, cenital_cones=data[1])
        agent_target['steer'] = self.horinzontal_control(ref_point=data[-2], img_center=img_center, img_base_len=image.shape[1])
        
        return agent_target, data

    def create_cone_map(self, centers, labels, eagle_img, image_shape, image):
        return self.cone_processing.create_cone_map(centers, labels, [eagle_img], orig_im_shape=image_shape, img_to_wrap=image)

    def horinzontal_control(self, ref_point, img_center, img_base_len):
        '''
        TODO MAKE IT WORK
        '''
        turn_point = img_center - ref_point
        # turn_point = -turn_point/img_base_len
        val = self.valTrackbarsPID()
        pid = self.pid_steer(Kp=val[0], Ki=val[1], Kd=val[2], setpoint=0, output_limits=(-1., 1.))
        return pid(turn_point)

    def longitudinal_control(self, agent_target, car_state, cenital_cones):
        blue_center, yell_center, oran_left_center, oran_rigth_center = cenital_cones

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

        agent_target = self.clutch_func(car_state, agent_target)

        # upgear, downgear, gear = self.change_gear(gear, rpm, throttle)
        return agent_target

'''# UNUSED
class AgentAccelerationYoloFast(AgentAcceleration):
    def __init__(self, logger, target_speed=20.):
        super().__init__(logger=logger)
        self.cone_processing = ConeProcessingNoWrapped()

    def get_action(self, agent_target, detections, speed, rpm, cone_centers=None, orig_im_shape=(1, 180, 320, 3), image=None):
        """
        Calcular los valores de control
        """
        bboxes, labels = detections

        data = self.cone_processing.create_cone_map(cone_centers, labels, None, orig_im_shape=(1,) + image.shape, img_to_wrap=image)

        img_center = int(image.shape[1] / 2)
        agent_target['steer'] = self.horinzontal_control(ref_point=data[-1], img_center=img_center, img_base_len=image.shape[1])

        agent_target = self.longitudinal_control(agent_target, cenital_cones=data[1], speed=speed, rpm=rpm)

        return agent_target, data

    def create_cone_map(self, centers, labels, eagle_img, image_shape, image):
        return self.cone_processing.create_cone_map(centers, labels, [eagle_img], orig_im_shape=image_shape, img_to_wrap=image)

    def horinzontal_control(self, ref_point, img_center, img_base_len):
        turn_point = img_center - ref_point
        # turn_point = -turn_point/img_base_len
        val = self.valTrackbarsPID()
        pid = self.pid_steer(Kp=val[0], Ki=val[1], Kd=val[2], setpoint=0, output_limits=(-1., 1.))
        return pid(turn_point)
        # return turn_point

    def longitudinal_control(self, agent_target, cenital_cones, speed, rpm):
        blue_center, yell_center, oran_left_center, oran_rigth_center = cenital_cones

        n_color_cones = len(blue_center) + len(yell_center)
        n_oran_cones = len(oran_left_center) + len(oran_rigth_center)

        val = self.valTrackbarsPID()
        pid_throttle = self.pid_throttle(Kp=val[3], Ki=val[4], Kd=val[5], setpoint=0, output_limits=(0., 1.))
        pid_brake = self.pid_brake(Kp=val[6], Ki=val[7], Kd=val[8], setpoint=0, output_limits=(0., 1.))

        if n_color_cones > n_oran_cones:
            target_speed = self.target_speed

            ref_point = speed - target_speed
            # throttle = pid_throttle(ref_point)
            agent_target['throttle'] = 1.0

            agent_target['brake'] = 0.
        else:
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

                ref_point = speed - target_speed
                # throttle = pid_throttle(ref_point)
                agent_target['throttle'] = 1.0
                agent_target['brake'] = 0.
            else: # Braking zone
                agent_target['throttle'] = 0.
                ref_point = -speed
                agent_target['brake'] = pid_brake(ref_point)

        # print(throttle, brake, ref_point, val[3], val[4], val[5], val[6], val[7], val[8])

        agent_target = self.clutch_func(speed, car_state, agent_target)

        return agent_target
'''