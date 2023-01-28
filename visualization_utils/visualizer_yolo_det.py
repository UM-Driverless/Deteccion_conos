import cv2
import numpy as np
from visualization_utils.visualize_base import VisualizeInterface
import time # Measure time taken

class Visualizer(VisualizeInterface):
    def __init__(self):
        self.saved_frames = []

    def visualize(self, agent_target, car_state, image, detections, fps, save_frames=False):
        """
        Main method that gets called to generate the visualization
        
        
        :param data: List of: [image, detections, cenital_map, y_hat, in_speed]
                     image: ndarray (h, w, channels)
                     detections: bboxes
                     cone_centers: liat of cone center coordinates [x, y]
                     cenital_map: eagle view image or coordinates map
                     in_speed: float. Current car speed
        :param controls: List of [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm]
                     throttle: float [0, 1]
                     brake: float [0, 1]
                     steer: float [-1, 1]
                     clutch: float [0, 1]
                     upgear: bool
                     downgear: bool
                     in_gear: int. current gear
                     in_rpm: float. current rpm
        :param fps: int
        :param save_frames: bool. Allows store the resulting frame in a list to later create a video with save_in_video function
        """
        self.make_images(agent_target, car_state, image, detections, fps, save_frames=save_frames)

    def make_images(self, agent_target, car_state, image, detections, fps, save_frames=False):
        '''Creates the new image with overlays of the detections.
        
        data:
        
        
        controls:
        
        fps: boolean with the current frames per second 
        save_frames: Whether to save the frames to a file, or just show them
        '''
        
        bbox, labels = detections
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Color values of each cone type, in bgr
        colors = {
            'blue_cone': (255, 0, 0),
            'yellow_cone': (0, 255, 255),
            'orange_cone': (40, 50, 200), #(40, 50, 200)
            'large_orange_cone': (40, 100, 255), #(40, 100, 255)
            'unknown_cone': (0,0,0)
        }

        # Old neural net
        # colors = [(255, 0, 0), (0, 255, 255), (100, 0, 255), (100, 0, 255)]
        
        # Print boxes around each detected cone
        image = self.print_bboxes(image, bbox, labels, colors)

        # Print cenital map
        # # image = self._print_cenital_map(cenital_map, colors, estimated_center, image) # TODO MAKE IT WORK

        # Print the output values of the agent, trying to control the car
        image = self.print_data(agent_target, car_state, fps, image, len(labels))

        # dim = (np.array(image.shape) * 0.1).astype('int')
        # image[400:400 + dim[1], 10:10 + dim[1]] = cv2.resize(wrap_img, (dim[1], dim[1]))

        #TODO TAKES 3ms OF TIME. make faster or in parallel
        cv2.imshow("Detections", image)
        cv2.waitKey(1)

        if save_frames:
            self.saved_frames.append(image)
        
        return image

    def print_data(self, agent_target, car_state, fps, image, cone_amount):
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
            f'    Cones: {cone_amount}',
            f'AGENT TARGET:',
            f'    throttle: {agent_target["throttle"]}',
            f'    brake: {agent_target["brake"]}',
            f'    steer: {agent_target["steer"]}',
            f'    clutch: {agent_target["clutch"]}',
        ]
        
        for i in range(11):
            image = cv2.putText(image, text[i], (10, 400+20*i), font, fontScale, color, thickness, cv2.LINE_AA)
        
        
        # Text from the agent. TODO it could be off. what then?
        # TODO FIGURE THIS OUT
        # ctr_img = self._controls_img(agent_target)
        # image[340:390, 10:210] = ctr_img
        
        return image

    '''
    def _print_cenital_map(self, cenital_map, color, estimated_center, image):
        cenital_img_size = 0.1
        # Pintar centros de masa de los conos
        dim = image.shape
        cenital_img = np.zeros((dim[1], dim[1], 3)) * 255
        # for c in cenital_map[0]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 20, color[0], -1)
        # for c in cenital_map[1]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 20, color[1], -1)
        for c in cenital_map[2]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 20, color[2], -1)
        for c in cenital_map[3]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 20, color[3], -1)

        cv2.circle(cenital_img, (int(estimated_center), int(dim[1] / 2)), 50, (0, 255, 0), -1)

        dim = (np.array(image.shape) * cenital_img_size).astype('int')
        image[2:2 + dim[1], 10:10 + dim[1]] = cv2.resize(cenital_img, (dim[1], dim[1]))

        return image
    '''

    def print_bboxes(self, image, bbox, label, colors):
        ''' Print bounding boxes around each detected cone
        
        '''
        
        for (box, lab) in zip(bbox, label):
            x1 = int(box[0, 0])
            y1 = int(box[0, 1])
            x2 = int(box[1, 0])
            y2 = int(box[1, 1])
            
            # NEWNET TODO
            # OLD
            # clase = int(lab[0]) # INT -> STR
            # lab = '{} {:.1f}'.format(lab[0], float(lab[1]))
            
            # NEW
            clase = str(lab[0]) # INT -> STR            
            lab = f'{lab[0]} {lab[1]}'
            
            # For bounding box
            image = cv2.rectangle(image, (x1, y1), (x2, y2), colors[clase], 2)
            
            # For the text background
            # Finds space required by the text so that we can put a background with that amount of width.
            # (w, h), _ = cv2.getTextSize(
            #     lab, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Prints the text.
            # image = cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), colors[clase], -1)

            # For printing text
            # TODO WE HAVE COMMENTED
            # image = cv2.putText(image, lab, (x1, y1),
            #                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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