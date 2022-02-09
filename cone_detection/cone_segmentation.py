import numpy as np
from ObjectDetectionSegmentation.Data.DataManager import DataManager

from cone_detection.detector_base import ConeDetectorInterface
import tensorflow as tf
import os
from tensorflow.python.saved_model import tag_constants
import cv2
# from Object


class ConeDetector(ConeDetectorInterface):

    def __init__(self, checkpoint_path="/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/ObjectDetectionSegmentation/DetectionData/SNet_3L_saved_model_FP32", logger=None):
        self.saved_model_loaded = tf.saved_model.load(checkpoint_path, tags=[tag_constants.SERVING])
        self.detection_model = self.saved_model_loaded.signatures['serving_default']

        self.logger = logger

    def saved_model(self):
        return self.saved_model_loaded, self.detection_model

    def detect_cones(self, input, dim=(1, 180, 320, 3), bbox=False, centers=False):
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

        all_labels = None
        if bbox:
            # Detectar bounding boxes
            all_bbox, all_labels = self.get_bbox(y_hat, _scale2original_size)
        else:
            all_bbox = None

        if centers:
            cone_centers, all_labels = self.get_centers(y_hat)
        else:
            cone_centers = None

        return [cone_centers, all_bbox, np.array(all_labels)], y_hat

    def get_bbox(self, y_hat, _scale2original_size):
        # Detectar bounding boxes
        all_labels = []
        all_bbox = []
        for i in range(y_hat.shape[0]):
            mask = np.argmax(y_hat[i], axis=2)
            labels = []
            bboxs = []
            for c in range(y_hat.shape[3] - 1):
                countours, _ = cv2.findContours(np.uint8((mask == c) * 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for countour in countours:
                    x, y, w, h = cv2.boundingRect(countour)
                    if w * h > 3:  # min_area
                        p1 = np.array([int(x * _scale2original_size[1]), int(y * _scale2original_size[0])])
                        p2 = np.array([int((x + w) * _scale2original_size[1]), int((y + h) * _scale2original_size[0])])
                        # bboxs.append(np.array([c, p1, p2]))
                        bboxs.append(np.array([p1, p2]))
                        labels.append(c)

            all_labels.append(np.array(labels))
            all_bbox.append(np.array(bboxs))
        return np.array(all_bbox)

    def get_centers(self, y_hat):
        # visualizar find contours
        # find contours in the binary image
        # y_hat_resize_backg = y_hat[0, :, :, 4]
        # aux = y_hat_resize_backg * 255
        # aux = aux.astype('uint8')
        # ret, foreground = cv2.threshold(aux, int(0.3 * 255), int(1. * 255), cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))


        all_labels = []
        all_cone_centers = []
        for i in range(y_hat.shape[0]):
            mask = np.argmax(y_hat[i], axis=2)
            # contours, hierarchy = cv2.findContours(foreground, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for clase in range(y_hat.shape[3] - 1):
                foreground = cv2.erode(np.uint8((mask == clase) * 1), kernel)
                contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                labels = []
                cone_centers = []
                for c in contours:
                    # calculate moments for each contour
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        # calculate x,y coordinate of center
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cone_centers.append([cX, cY])
                        labels.append(clase)
                all_labels.append(np.array(labels))
                all_cone_centers.append((np.array(cone_centers)))

        return np.array(all_cone_centers), np.array(all_labels)
