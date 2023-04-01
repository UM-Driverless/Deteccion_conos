import os
import numpy as np
import math
from skimage.io import imread, imshow
from skimage import transform
import matplotlib.pyplot as plt
import torch
import cv2
print(f'IS CUDA AVAILABLE? : {torch.cuda.is_available()}')

from cone_detection.detector_base import ConeDetectorInterface
from globals.globals import * # Global variables and constants, as if they were here

class ConeDetector(ConeDetectorInterface):
    def __init__(self, checkpoint_path="yolov5/weights/yolov5_models/best.pt", logger=None):
        self.checkpoint_path = checkpoint_path
        print(f'Using weights in: {checkpoint_path}')
        self.detection_model = torch.hub.load('yolov5/', 'custom', path=checkpoint_path, source='local', force_reload=True)
        self.detection_model.conf = 0.3
        self.logger = logger
        

    def detect_cones(self, image):
        """
        Takes image as array [[ [b,g,r],...],...], and returns an array with the bounding boxes (corners) and labels of the detected cones, and the cone_centers separately
        
        
        Performs the cones detection task. The detection must include the bounding boxes and classification of each
        cone.
        :param img: (3D numpy array) Image to process.
        :param get_centers: Bool_ If True calculates the cone centers.
        :return: [[ndarray, ndarray], list] ndarray with detected bounding boxes and classification of each cone, list of auxiliar
                                data, in this case segmentations.
        """
        
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
        
        xmax_coords = [row['xmax'] for _, row in results.pandas().xyxy[0].iterrows()]
        ymax_coords = [row['ymax'] for _, row in results.pandas().xyxy[0].iterrows()]
        
        # Intento de vista cenital con warp, no sale bien
        # cone_coords = np.column_stack((xmax_coords, ymax_coords)).astype(np.float32)
        # # cone_coords = np.array([xmax_coords, ymax_coords], dtype=np.float32)
        
        # area_of_interest = np.array([[300, 200], [340, 200], [640, 640], [0, 640]], dtype=np.float32)
        # area_of_projection = np.array([[0, 0], [640, 0], [400, 640], [240, 640]], dtype=np.float32)
        # H, _ = cv2.findHomography(area_of_interest, area_of_projection)
        # # Reshape the cone_coords array to be a (num_cones, 1, 2) array
        # cone_coords_3d = cone_coords.reshape(-1, 1, 2)

        # # Apply the homography matrix to the cone coordinates
        # cone_coords_transformed_3d = cv2.perspectiveTransform(cone_coords_3d, H)

        # # Reshape the transformed coordinates to a (num_cones, 2) array
        # cone_coords_transformed = cone_coords_transformed_3d.reshape(-1, 2) # -1 means unknown, let numpy figure it out
        
        # # Extract the xmax and ymax coordinates from the transformed coordinates
        # xmax_coords = cone_coords_transformed[:, 0]
        # ymax_coords = cone_coords_transformed[:, 1]
        
        for _, row in results.pandas().xyxy[0].iterrows():
            # Filter how many cones we want to use, according to the confidence value (0 to 1)
            if float(row['confidence']) > CONFIDENCE_THRESHOLD:                
                cones.append([ # TODO ARRAY OF DICTIONARIES INSTEAD?
                            str(row['name'].split('class')[-1]), # label
                            [[int(row['xmin']), int(row['ymin'])],[int(row['xmax']), int(row['ymax'])]], # bbox
                            # [(row['ymax']), (row['ymin'])/2], # coords x,y
                            [xmax_coords[_], ymax_coords[_]],
                            row['confidence']
                            ])
        
        return cones