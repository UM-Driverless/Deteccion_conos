import numpy as np
import torch
import cv2
from cone_detection.detector_base import ConeDetectorInterface

class ConeDetector(ConeDetectorInterface):

    def __init__(self, checkpoint_path="yolov5/weights/yolov5s-cones-mixed-classes/weights/best.pt", logger=None):
        self.checkpoint_path = checkpoint_path
        self.detection_model = torch.hub.load('yolov5/', 'custom', path=checkpoint_path, source='local', force_reload=True)
        self.logger = logger

    def detect_cones(self, input, get_centers=False):
        """
        Performs the cones detection task. The detection must include the bounding boxes and classification of each
        cone.
        :param img: (3D numpy array) Image to process.
        :param get_centers: Bool_ If True calculates the cone centers.
        :return: [[ndarray, ndarray], list] ndarray with detected bounding boxes and classification of each cone, list of auxiliar
                                data, in this case segmentations.
        """
        results = self.detection_model(input)
        bboxes = []
        labels = []
        # self.extract_data(results)
        for index, row in results.pandas().xyxy[0].iterrows():
            x1 = int(row['xmin'])
            x2 = int(row['xmax'])
            y1 = int(row['ymin'])
            y2 = int(row['ymax'])
            label = [int(row['name']), float(row['confidence'])]
            # x1,y1------------
            # |               |
            # |               |
            # |               |
            # |               |
            # ------------X2,Y2
            bboxes.append([[x1, y1], [x2, y2]])
            labels.append(label)

        if get_centers:
            cone_centers = self.get_centers(bboxes, labels)
        else:
            cone_centers = None

        return [np.array(bboxes), np.array(labels)], np.array(cone_centers)


    def get_centers(self, bboxes, labels):
        bboxes = np.array(bboxes)
        x_center = bboxes[:, 0, 0] + ((bboxes[:, 1, 0] - bboxes[:, 0, 0]) / 2).astype('int')
        y_center = bboxes[:, 0, 1] + ((bboxes[:, 1, 1] - bboxes[:, 0, 1]) / 2).astype('int')

        cone_centers = np.array([x_center, y_center]).transpose()

        index_0 = np.array(labels)[:, 0] == 0
        index_1 = np.array(labels)[:, 0] == 1
        index_2 = np.array(labels)[:, 0] == 2
        index_3 = np.array(labels)[:, 0] == 3  # Los naranjas están todos agrupados en una única clase, por lo que este indez_4 no hace falta

        cone_class_0 = cone_centers[index_0]
        cone_class_1 = cone_centers[index_1]
        cone_class_2 = cone_centers[index_2]
        cone_class_3 = cone_centers[index_3]
        cone_centers_by_class = [cone_class_0, cone_class_1, cone_class_2, cone_class_3]

        return np.array(cone_centers_by_class)
