import numpy as np
import torch
print(f'IS CUDA AVAILABLE? : {torch.cuda.is_available()}')

from cone_detection.detector_base import ConeDetectorInterface
from globals.globals import * # Global variables and constants, as if they were here

class ConeDetector(ConeDetectorInterface):
    # TODO FALTA LLAMAR AL INIT
    def __init__(self, checkpoint_path="yolov5/weights/yolov5_models/best.pt", logger=None):
        self.checkpoint_path = checkpoint_path
        print(f'Using weights in: {checkpoint_path}')
        self.detection_model = torch.hub.load('yolov5/', 'custom', path=checkpoint_path, source='local', force_reload=True)
        self.detection_model.conf = 0.3
        self.logger = logger
        

    def detect_cones(self, image, get_centers=False):
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
        
        
        bboxes = [] # min and max corners
        labels = [] # [{class name}, {confidence}]
        # self.extract_data(results)
        
        for _, row in results.pandas().xyxy[0].iterrows():
            # Filter how many cones we want to use, according to the confidence value (0 to 1)
            if float(row['confidence']) > CONFIDENCE_THRESHOLD:
                # Add bounding box: [ (min)[x,y], (max)[x,y] ]
                bboxes.append([[int(row['xmin']), int(row['ymin'])], [int(row['xmax']), int(row['ymax'])]])
                
                # Add label: [{CLASS NAME string}, {CONFIDENCE float number f32}]
                labels.append([str(row['name'].split('class')[-1]), float(row['confidence'])])
                # Old neural net, with class names 0 1 2 3...
                # labels.append([int(row['name'].split('class')[-1]), float(row['confidence'])])

        if get_centers:
            cone_centers = self.get_centers(np.array(bboxes), labels)
        else:
            cone_centers = None

        return [np.array(bboxes), np.array(labels)], np.array(cone_centers)


    def get_centers(self, bboxes, labels):
        '''
        Returns a list with the cone centers for each class: [ [[x,y],[x,y],...], [[x,y],[x,y],...], ...]
        Takes the bboxes array with the position of all cones, and the labels linked with each cone.
        
        bboxes is a list of cone positions, where each center is a list of 2 points, where each point is a list of 2 [x,y] coordinates.
        labels is a list of [{CLASS NAME},{CONFIDENCE}] pairs.
        
        '''
        
        if bboxes.shape[0] > 0:
            # TODO check can delete this
            # x_center = bboxes[:, 0, 0] + ((bboxes[:, 1, 0] - bboxes[:, 0, 0]) / 2).astype('int')
            # y_center = bboxes[:, 0, 1] + ((bboxes[:, 1, 1] - bboxes[:, 0, 1]) / 2).astype('int')

            x_center = ((bboxes[:, 0, 0] + bboxes[:, 1, 0]) / 2).astype('int')
            y_center = ((bboxes[:, 0, 1] + bboxes[:, 1, 1]) / 2).astype('int')

            # list of [x,y] coordinates of all cones
            cone_centers = np.array([x_center, y_center]).transpose()

            # Old neural net had 0 1 2 3 numbers instead of 'blue_cone' etc strings
            # Boolean, is blue, is yellow, is orange...
            cones_blue = np.array(labels)[:, 0] == 'blue_cone'
            cones_yellow = np.array(labels)[:, 0] == 'yellow_cone'
            cones_orange = np.array(labels)[:, 0] == 'orange_cone'
            
            cones_large = np.array(labels)[:, 0] == 'large_orange_cone'  # Los naranjas están todos agrupados en una única clase TODO delete or not use?
            cones_unknown = np.array(labels)[:, 0] == 'unknown_cone'

            # [ [(blue)[x,y],[x,y],...], [(yellow)[x,y],[x,y],...], ...]
            cone_centers_by_class = [cone_centers[cones_blue], 
                                     cone_centers[cones_yellow], 
                                     cone_centers[cones_orange], 
                                     cone_centers[cones_large], 
                                     cone_centers[cones_unknown]]

            return np.array(cone_centers_by_class, dtype=object) # dtype=object makes array a list of pointers instead of store the objects themselves
        else:
            return np.array([[], [], [], [], []])
