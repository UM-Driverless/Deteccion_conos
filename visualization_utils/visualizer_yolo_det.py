import cv2
import numpy as np
import time # Measure time taken
from globals.globals import * # Global variables and constants, as if they were here

class Visualizer():
    def __init__(self):
        self.saved_frames = []
        
        # Color values of each cone type, in bgr
        self.colors = {
            'blue_cone': (255, 0, 0),
            'yellow_cone': (0, 255, 255),
            'orange_cone': (40, 50, 200), #(40, 50, 200)
            'large_orange_cone': (40, 100, 255), #(40, 100, 255)
            'unknown_cone': (0,0,0)
        }

    def visualize(self, agent_target, car_state, image, cones, fps, save_frames=False):
        """
        Generates the visualization
        
        ...
        save_frames: Whether to save the frames to a file, or just show them
        
        
        """
        image = self.make_image(agent_target, car_state, image, cones, fps)
        
        # Show and save image
        cv2.imshow("Detections", image)
        cv2.waitKey(1)

        if save_frames:
            self.saved_frames.append(image)

    def make_image(self, agent_target, car_state, image, cones, fps):
        '''Creates the new image with overlays of the detections.
        
        
        
        fps: boolean with the current frames per second
        '''
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Print boxes around each detected cone
        image = self._print_bboxes(image, cones)

        # Add text with distance (x,y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.4
        color = (255, 255, 255)
        thickness = 1
        for cone in cones:
            x_mid = int((cone['bbox'][0][0] + cone['bbox'][1][0])/2)
            y_bottom = int(cone['bbox'][0][1])
            # Print x distance in m
            image = cv2.putText(image, f"({cone['coords']['x']:3.1f},{cone['coords']['y']:3.1f})", (x_mid, y_bottom), font, fontScale, color, thickness, cv2.LINE_AA)
        
        # Print cenital map
        image = self._print_cenital_map(image, cones)

        # Print the output values of the agent, trying to control the car
        image = self._print_data(cones, agent_target, car_state, fps, image)

        # dim = (np.array(image.shape) * 0.1).astype('int')
        # image[400:400 + dim[1], 10:10 + dim[1]] = cv2.resize(wrap_img, (dim[1], dim[1]))
        
        #TODO TAKES 3ms OF TIME. make faster or in parallel
        
        return image

    def _print_data(self, cones, agent_target, car_state, fps, image):
        # config
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        # Dark background
        background = np.zeros_like(image)
        cv2.rectangle(background, (0,380), (640, 640), (1,1,1), cv2.FILLED)
        alpha = 0.5
        mask = background.astype(bool)
        image[mask] = cv2.addWeighted(src1=image, alpha=alpha, src2=background, beta=(1-alpha), gamma=0)[mask]
        
        # Add text (takes almost no time to run)
        text = [
            f'CAR STATE:',
            f'    Speed: {car_state["speed"]:.2f}',
            f'    RPM: {int(car_state["rpm"]):d}',
            f'NET DATA:',
            f'    fps: {int(fps):.2f}',
            f'    Cones: {len(cones)}',
            f'AGENT TARGET:',
            f'    acc: {agent_target["acc"]}',
            f'    steer: {agent_target["steer"]}',
            f'    throttle: {agent_target["throttle"]}',
            f'    brake: {agent_target["brake"]}',
            f'    clutch: {agent_target["clutch"]}',
        ]
        
        for i in range(11):
            image = cv2.putText(image, text[i], (10, 400+20*i), font, fontScale, color, thickness, cv2.LINE_AA)
        
        
        # Text from the agent. TODO it could be off. what then?
        # TODO FIGURE THIS OUT
        # ctr_img = self._controls_img(agent_target)
        # image[340:390, 10:210] = ctr_img
        
        return image

    def _print_cenital_map(self, image, cones):
        '''
        Returns the image with cones represented as circles from a previously-calculated top view.
        
        Takes:
            image: The original image, to add the cenital map to it
            cenital_map: The coordinates of the cones to show, as an array
                = [{array of blue cones},{array of yellow cones},...]
                = [[[x,y],[x,y],[x,y],...],[[x,y],[x,y],[x,y],...],...]
        
        The size of the image is configured using the x coordinate, horizontal
        '''
        img_size = image.shape[1] # vertical size in pixels
        
        cenit_perc = VISUALIZER_CENITAL_MAP_SIZE_PERC
        cenital_size = int(img_size * cenit_perc)
        cenital_img = np.zeros((img_size, img_size, 3))
        
        for cone in cones:
            # cenital_img = cv2.circle(cenital_img, (int(cone['coords']['x']*cenit_perc), int(cone['coords']['y']*cenit_perc)), 4, self.colors[cone['label']], -1)
            cenital_img = cv2.circle(cenital_img, (int(320 + 640/48 * cone['coords']['y']), int(640 - 640/48 * cone['coords']['x'])), 4, self.colors[cone['label']], -1)
            
        image[0:cenital_size, 0:cenital_size] = cv2.resize(cenital_img, (cenital_size, cenital_size))

        return image

    def _print_bboxes(self, image, cones):
        ''' Print bounding boxes around each detected cone
        
        '''
        
        for cone in cones:
            image = cv2.rectangle(image, (int(cone['bbox'][0][0]), int(cone['bbox'][0][1])), (int(cone['bbox'][1][0]), int(cone['bbox'][1][1])), self.colors[cone['label']], 1)
            
        return image

    def _controls_img(self, agent_target):
        steer = agent_target['steer']
        throttle = agent_target['throttle']
        brake = agent_target['brake']
        clutch = agent_target['clutch']
        
        text_steer =    'steer:  {:+.3f}'.format(steer)
        text_throttle = 'throttle: {:.3f}'.format(throttle)
        text_brake =    'brake:   {:.3f}'.format(brake)
        text_clutch =   'clutch:  {:.3f}'.format(clutch)

        ste_img = np.ones((5, 41, 3), dtype=np.uint8) * 255
        thr_img = np.ones((5, 41, 3), dtype=np.uint8) * 255
        brk_img = np.ones((5, 41, 3), dtype=np.uint8) * 255
        clutch_img = np.ones((5, 41, 3), dtype=np.uint8) * 255

        ctr_img = np.zeros((36, 87, 3), dtype=np.uint8)

        steer = np.int((steer + 1) / 2 * 41)
        throttle = np.int(np.clip(throttle, 0.0, 1.0) * 41)
        brake = np.int(np.clip(brake, 0.0, 1.0) * 41)
        # brake = np.int(np.clip(brake, 0.0, 1.0) * 41)
        clutch = np.int(np.clip(clutch, 0.0, 1.0) * 41)

        ste_img[:, steer:steer + 1, 1:3] = np.zeros((5, 1, 2), dtype=np.uint8)
        thr_img[:, :throttle, 1] = thr_img[:, :throttle, 1] * 0
        brk_img[:, :brake, 2] = brk_img[:, :brake, 2] * 0
        clutch_img[:, :clutch, 0] = clutch_img[:, :clutch, 0] * 0

        ctr_img[3:8, 43:84, :] = ste_img
        ctr_img[12:17, 43:84, :] = thr_img
        ctr_img[20:25, 43:84, :] = brk_img
        ctr_img[28:33, 43:84, :] = clutch_img


        ctr_img = cv2.resize(ctr_img, (200, 50))

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.4
        color = (255, 255, 255)
        thickness = 1
        ctr_img = cv2.putText(ctr_img, text_steer, (1, 10), font, fontScale, color, thickness, cv2.LINE_AA)
        ctr_img = cv2.putText(ctr_img, text_throttle, (1, 22), font, fontScale, color, thickness, cv2.LINE_AA)
        ctr_img = cv2.putText(ctr_img, text_brake, (1, 34), font, fontScale, color, thickness, cv2.LINE_AA)
        ctr_img = cv2.putText(ctr_img, text_clutch, (1, 46), font, fontScale, color, thickness, cv2.LINE_AA)

        return ctr_img

    def save_in_video(self, path, name):
        for i in range(len(self.saved_frames)):
            cv2.imwrite(path + name.format(i), self.saved_frames[i])
        cv2.destroyAllWindows()