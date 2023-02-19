# IMPORTS
# import numpy as np
# import matplotlib.pyplot as plt # For representation of time consumed
# from canlib import canlib, Frame

## Our imports
from globals.globals import * # Global variables and constants, as if they were here


# FUNCTIONS

        
''' Visualize thread doesn't work. It's not required for the car to work so ignore it.
# def visualize_thread():
#     print(f'Starting visualize thread...')
#     global visualizer
#     global image, detections, cone_centers, cenital_map, in_speed
#     global throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm, fps
    
#     print(visualizer)
#     visualizer.visualize([image, detections, cone_centers, cenital_map, in_speed],
#                         [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm], fps,
#                         save_frames=False)
'''

'''
# Visualize thread directly here
def visualize_thread():
    print(f'Starting visualize thread...')
    while True:        
        # image, detections, cone_centers, cenital_map, speed = [image, detections, cone_centers,cenital_map, in_speed]
        bbox, labels = detections
        # cenital_map, estimated_center, wrap_img = cenital_map
        # throttle, brake, steer, clutch, upgear, downgear, gear, rpm = [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm]
        
        image = cam_queue.get(block=False, timeout=5) # Read an image but don't remove it. Only the main loop takes it
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Color values of each cone type, in bgr
        colors = {
            'blue_cone': (255, 0, 0),
            'yellow_cone': (0, 255, 255),
            'orange_cone': (40, 50, 200), #(40, 50, 200)
            'large_orange_cone': (40, 100, 255), #(40, 100, 255)
            'unknown_cone': (0,0,0)
        }
        
        # Print boxes around each detected cone
        image = Visualizer.print_bboxes(image, bbox, labels, colors)

        # Print cenital map
        # image = self._print_cenital_map(cenital_map, colors, estimated_center, image) # TODO MAKE IT WORK

        # Print the output values of the agent, trying to control the car
        # image = Visualizer.print_data(0, 0, fps, 0, image, 0, 0, 0, 0, len(labels))

        

        # dim = (np.array(image.shape) * 0.1).astype('int')
        # image[400:400 + dim[1], 10:10 + dim[1]] = cv2.resize(wrap_img, (dim[1], dim[1]))

        #TODO make faster or in parallel #takestime
        cv2.imshow("Detections", image)
        cv2.waitKey(100)
'''
