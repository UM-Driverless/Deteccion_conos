import numpy as np
from ObjectDetectionSegmentation.Data.DataManager import DataManager

from cone_detection.detector_base import ConeDetectorInterface
import tensorflow as tf
import os
from tensorflow.python.saved_model import tag_constants
import cv2
# from Object



class ConeDetector(ConeDetectorInterface):

    def __init__(self, checkpoint_path="/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/ObjectDetectionSegmentation/DetectionData/SNet_3L_saved_model_FP32"):
        self.saved_model_loaded = tf.saved_model.load(checkpoint_path, tags=[tag_constants.SERVING])
        self.detection_model = self.saved_model_loaded.signatures['serving_default']
    def saved_model(self):
        return self.saved_model_loaded, self.detection_model

    def detect_cones(self, input, dim=(1, 180, 320, 3)):
        """
        Performs the cones detection task. The detection must include the bounding boxes and classification of each
        cone.
        :param img: (3D numpy array) Image to process.
        :param min_score_thresh: (float in [0., 1.]) Min score of confident on a detection.
        :param show_detections: (bool) If True: The image with detections id displayed. If False: no image is displayed.
        :param im_name: (string) Name of the detection image when show_detections=True
        :return: [ndarray, list] ndarray with detected bounding boxes and classification of each cone, list of auxiliar
                                data, in this case segmentations.
        """
        im_size = input.shape[:2]
        img_yuv = cv2.cvtColor(input, cv2.COLOR_RGB2YUV)
        img_yuv = cv2.resize(img_yuv, (320, 180)) / 255.
        input = tf.constant(np.expand_dims(img_yuv, 0), dtype=tf.float32)
        _scale2original_size = [im_size[0] / dim[1], im_size[1] / dim[2]]
        y_hat = self.detection_model(input)
        y_hat = y_hat["conv2d_7"]
        y_hat = y_hat.numpy()

        all_bboxs = []
        for i in range(y_hat.shape[0]):
            mask = np.argmax(y_hat[i], axis=2)
            bboxs = []
            for c in range(y_hat.shape[3] - 1):
                countours, _ = cv2.findContours(np.uint8((mask == c) * 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for countour in countours:
                    x, y, w, h = cv2.boundingRect(countour)
                    if w * h > 3:  # min_area
                        p1 = [int(x * _scale2original_size[1]), int(y * _scale2original_size[0])]
                        p2 = [int((x + w) * _scale2original_size[1]), int((y + h) * _scale2original_size[0])]
                        bboxs.append([c, p1, p2])
            all_bboxs.append(bboxs)

        return all_bboxs, y_hat
