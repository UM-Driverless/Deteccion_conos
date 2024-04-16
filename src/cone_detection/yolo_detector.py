import os, sys, math, cv2, yaml
import numpy as np
from skimage.io import imread, imshow
from skimage import transform
import matplotlib.pyplot as plt
import torch
print(f'IS CUDA AVAILABLE? : {torch.cuda.is_available()}')

class ConeDetector():
    def __init__(self, checkpoint_path="~/Deteccion_conos/yolov5/weights/yolov5_models/best_adri.pt"):
        self.checkpoint_path = checkpoint_path
        print(f'Using weights in: {checkpoint_path}')
        self.detection_model = torch.hub.load('yolov5/', 'custom', path=checkpoint_path, source='local', force_reload=True)
        self.detection_model.conf = 0.3

    def detect_cones(self, image, car_state):
        """
        Takes image as array [[ [b,g,r],...],...], and returns an array with the bounding boxes (corners) and labels of the detected cones, and the cone_centers separately
        
        
        Performs the cones detection task. The detection must include the bounding boxes and classification of each
        cone.
        :param img: (3D numpy array) Image to process.
        :param get_centers: Bool_ If True calculates the cone centers.
        :return: [[ndarray, ndarray], list] ndarray with detected bounding boxes and classification of each cone, list of auxiliar
                                data, in this case segmentations.
        """
        # CAM CONSTANTS
        CAMERA_VERTICAL_FOV_DEG = 90 # ZED horizontal FOV: 120
        CAM_HEIGHT = 0.75 # m
        CAM_HORIZON_POS = 0.5 # per 1 of image from top
        # Simulator camera (cam1) pos: (-.3,-.16,.8)
        CONFIDENCE_THRESHOLD = 0.5 # 0.5
        
        
        # NEURAL NETWORK DETECTION
        results = self.detection_model(image) # If there are no cones, results will be empty
        
        # Create a 'cones' array with each cone being [{label},{bbox},{coords},{confidence}]
            # {label} is a string that defines the type of cone
            # {bbox} is [{corner_min},{corner_max}]
                # corner min is top left, corner max is bottom right
                # each position is [x,y]
                # x goes right and y down in the image
            # {coords} is [x,y] of the cone respect to the car, x goes forwards and y left.
            # {confidence} is part of the net
        cones = []
        
        # x_coords = [(row['xmax']+row['xmin'])/2 for _, row in results.pandas().xyxy[0].iterrows()]
        #ymax_coords = [row['ymax'] for _, row in results.pandas().xyxy[0].iterrows()]
        
        
        # To calibrate camera:
            # Field of view (angle)
            # Horizon position in camera view
            # Camera height
        
        image_height_px = 640
        f = 2.8e-3 #m # does not affect
        
        # MAIN CONFIGURATION DATA
        XZ_FOV_Rad = CAMERA_VERTICAL_FOV_DEG * math.pi / 180
        
        horizon_px_from_top = image_height_px * CAM_HORIZON_POS # horizon at 50% from top. HUGE EFFECT.
        
        vertical_pix_to_rad = XZ_FOV_Rad / image_height_px
        pixel_size = f * math.tan(XZ_FOV_Rad/2) * 2 / image_height_px# m
        for i, row in results.pandas().xyxy[0].iterrows():
            # Filter how many cones we want to use, according to the confidence value (0 to 1)
            if row['confidence'] > CONFIDENCE_THRESHOLD:
                # Distance using cone base height in camera
                projected_height_from_top = f * math.tan((row['ymax'] - horizon_px_from_top) * vertical_pix_to_rad)
                projected_lateral = ((row['xmax']+row['xmin'])/2 - image_height_px/2) * pixel_size # m
                distance_horiz = f * CAM_HEIGHT / projected_height_from_top
                
                angle_z = math.atan(projected_lateral / f)
                
                # x is longitudinal, y is lateral
                longit = distance_horiz * math.cos(angle_z)
                lateral = -distance_horiz * math.sin(angle_z)
                
                cones.append({
                    'label': str(row['name'].split('class')[-1]),
                    'bbox': [[row['xmin'], row['ymin']],[row['xmax'], row['ymax']]],
                    'coords': {
                        # 'x': (row['xmax']+row['xmin'])/2, #x_coords[i], # X_world[i,0],
                        # 'y': row['ymax'] #ymax_coords[i] #X_world[i,1]
                        'x': longit, # m
                        'y': lateral, # m
                    },
                    'confidence': row['confidence']
                })
        
        return cones