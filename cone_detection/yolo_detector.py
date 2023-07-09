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
# import tools.tools as tools



class ConeDetector(ConeDetectorInterface):
    def __init__(self, checkpoint_path="yolov5/weights/yolov5_models/best.pt"):
        self.checkpoint_path = checkpoint_path
        print(f'Using weights in: {checkpoint_path}')
        self.detection_model = torch.hub.load('yolov5/', 'custom', path=checkpoint_path, source='local', force_reload=True)
        self.detection_model.conf = 0.3

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
        
        # x_coords = [(row['xmax']+row['xmin'])/2 for _, row in results.pandas().xyxy[0].iterrows()]
        #ymax_coords = [row['ymax'] for _, row in results.pandas().xyxy[0].iterrows()]
        
        '''
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
        '''
        
        '''
        # INTENTO DE TRIANGULACIÓN. LO DEJO COMO REFERENCIA HASTA QUE TENGAMOS ALGO QUE FUNCIONE BIEN
        # Define intrinsic camera parameters
        K = np.array([[500, 0, 320], [0, 500, 320], [0, 0, 1]])

        # Define extrinsic camera parameters
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity rotation matrix
        t = np.array([0., 0., 1.0])  # Camera position in world coordinates

        # Define image coordinates of the cones
        img_pts = np.array([[x_coords[i], ymax_coords[i]] for i in range(len(x_coords))])

        # Define a dummy set of image coordinates for the second camera
        img_pts_2 = img_pts+0.1 # por poner algo de diferencia
        R_2 = np.array([[0, 0, -1], [0, 1, 0], [0, 1, 0]])  # Identity rotation matrix
        t_2 = np.array([0., 0., 1.])  # Camera position in world coordinates
        
        # Compute the projection matrices
        P = np.dot(K, np.hstack((R, t.reshape(3, 1))))
        P_2 = np.dot(K, np.hstack((R_2, t_2.reshape(3, 1))))
        
        # Convert image coordinates to homogeneous coordinates
        img_pts_h = np.hstack((img_pts, np.ones((img_pts.shape[0], 1))))
        img_pts_2_h = np.hstack((img_pts_2, np.ones((img_pts_2.shape[0], 1))))
        
        # Triangulate the 3D coordinates of the cones
        X_h = cv2.triangulatePoints(P, P_2, img_pts.T, img_pts_2.T).T
        X = X_h[:, :3] / X_h[:, 3:]

        # Convert to real-world coordinates
        X_world = np.dot(R.T, X.T - t.reshape(3, 1)).T
        '''
        
        # To calibrate camera:
            # Field of view (angle)
            # Horizon position in camera view
            # Camera height
        
        image_size_px = 640
        f = 2.8e-3 #m # does not affect
        
        # MAIN CONFIGURATION DATA
        FOV_Rad = CAMERA_VERTICAL_FOV_DEG * math.pi / 180
        horizon_px_from_top = image_size_px * CAM_HORIZON_POS # horizon at 50% from top. HUGE EFFECT.
        
        pix_to_rad = FOV_Rad / image_size_px
        pixel_size = f * math.tan(FOV_Rad/2) * 2 / image_size_px# m
        for i, row in results.pandas().xyxy[0].iterrows():
            # Filter how many cones we want to use, according to the confidence value (0 to 1)
            if row['confidence'] > CONFIDENCE_THRESHOLD:                
                # Distance using cone base height in camera
                projected_height_from_top = f * math.tan((row['ymax'] - horizon_px_from_top) * pix_to_rad)
                projected_lateral = ((row['xmax']+row['xmin'])/2 - image_size_px/2) * pixel_size # m
                distance_horiz = f * CAM_HEIGHT / projected_height_from_top
                
                angle_z = math.atan(projected_lateral / f)
                
                # x is longitudinal, y is lateral
                longit = distance_horiz * math.cos(angle_z)
                lateral = distance_horiz * math.sin(angle_z)
                
                cones.append({
                    'label': str(row['name'].split('class')[-1]),
                    'bbox': [[row['xmin'], row['ymin']],[row['xmax'], row['ymax']]],
                    'coords': { # TODO DO THE RIGHT CALCULATION HERE
                        # 'x': (row['xmax']+row['xmin'])/2, #x_coords[i], # X_world[i,0],
                        # 'y': row['ymax'] #ymax_coords[i] #X_world[i,1]
                        'x': longit, # m
                        'y': lateral, # m
                    },
                    'confidence': row['confidence']
                })
        
        return cones