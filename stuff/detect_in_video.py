import cv2
import numpy as np
from connection_utils.my_client import ConnectionManager
from agent.agent_v0 import Agent
from cone_detection.cone_segmentation import ConeDetector
from trajectory_estimation.cone_processing import ConeProcessing
from visualization_utils.visualize import Visualize
import tensorflow as tf
from ObjectDetectionSegmentation.Data.DataManager import DataManager
from ObjectDetectionSegmentation.Networks import ModelManager
import os
import matplotlib.pyplot as plt
import time
import pygame

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ["CUDA_DEVICE_ORDER"] = '0'

# Estas tres lineas resuelven algunos problemas con cuDNN en TF2 por los que no me permitía ejecutar en GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def _bbox_inference4seg(input, original_size, dim, model):
    _scale2original_size = [original_size[0] / dim[1], original_size[1] / dim[2]]
    y_hat = model(input)
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

if __name__ == '__main__':
    image_folder = '/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/video/'

    im_name = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    # im_name.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    im_name.sort()
    im_name = im_name[:662]
    frame = cv2.imread(os.path.join(image_folder, im_name[0]))
    height, width, layers = frame.shape


    verbose = 1

    # Inicializar detector de conos
    # detector = ConeDetector()

    from tensorflow.python.saved_model import tag_constants

    saved_model_loaded = tf.saved_model.load(
        "/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/ObjectDetectionSegmentation/DetectionData/SNet_3L_saved_model_FP32",
        tags=[tag_constants.SERVING])
    detection_model = saved_model_loaded.signatures['serving_default']

    path_images = "/home/shernandez/PycharmProjects/UMotorsport/v1/Object-Detection-with-Segmentation-main/DetectionData/26_02_2021__16_59_0"
    path_images_destination = "/home/shernandez/PycharmProjects/UMotorsport/v1/Object-Detection-with-Segmentation-main/DetectionData/images_dest"
    limit = 1  # <============================= unlabeled image limit
    color_space = 82  # <====== bgr=None, lab=44, yuv=82, hsv=40, hsl=52
    input_dims = (1, 180, 320, 3)
    unlabed, names, sizes = DataManager.loadUnlabeled(path_images, path_images_destination, limit, color_space,
                                                      input_dims[1:3])
    # y_hat = detection_model(tf.constant(unlabed))

    # bboxes = _bbox_inference4seg(tf.constant(unlabed), (1280, 720), input_dims, detection_model)

    # cone_processing = ConeProcessing()
    # Inicializar Agente (controlador)
    agent = Agent()
    # Inicializar conexiones
    # connect_mng = ConnectionManager()
    # Visualización de datos
    visualizator = Visualize()

    video = []
    try:
        for i in range(len(im_name)):
            start_time = time.time()

            # Pedir datos al simulador o al coche
            image = cv2.imread(os.path.join(image_folder, im_name[i]))
            # cv2.imshow("bgr", image)

            # Detectar conos
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            img_yuv = cv2.resize(img_yuv, (320, 180)) / 255.
            detections, y_hat = _bbox_inference4seg(tf.constant(np.expand_dims(img_yuv, 0), dtype=tf.float32), image.shape[:2], input_dims, detection_model)

            color_mask = np.argmax(y_hat[0], axis=2)

            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("rgb", image)
            y_hat_resize_blue = y_hat[0, :, :, 0]
            y_hat_resize_yell = y_hat[0, :, :, 1]
            y_hat_resize_oran = y_hat[0, :, :, 2]
            y_hat_resize_boran = y_hat[0, :, :, 3]
            y_hat_resize_backg = y_hat[0, :, :, 4]


            res_image = np.zeros((180, 320, 3), dtype=np.uint8)

            res_image[:, :, 0][color_mask == 0] = 255

            res_image[:, :, 1][color_mask == 1] = 255
            res_image[:, :, 2][color_mask == 1] = 255

            res_image[:, :, 1][color_mask == 2] = 130
            res_image[:, :, 2][color_mask == 2] = 255

            res_image[:, :, 1][color_mask == 3] = 130
            res_image[:, :, 2][color_mask == 3] = 255

            cv2.imshow("color cones cones", res_image)
            cv2.imshow("background", y_hat_resize_backg)

            for det in detections[0]:
                if det[0] == 0:
                    color = (255, 0, 0)
                elif det[0] == 1:
                    color = (0, 255, 255)
                elif det[0] == 2:
                    color = (102, 178, 255)
                else:
                    color = (102, 178, 255)
                    # color = (0, 76, 153)
                a = tuple(det[1])
                b = tuple(det[2])
                image = cv2.rectangle(image, a, b, color=color, thickness=3)

            image[20:200, 10:330, :] = res_image
            cv2.imshow("cones", image)
            cv2.waitKey(1)
            video.append(image)

            print("FPS: ", 1.0 / (time.time() - start_time))

    finally:
        # out = cv2.VideoWriter(
            # '/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/cone_detection_FS.avi',
            # -1, 30.0, (height, width))
        # out = cv2.VideoWriter('/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/cone_detection_FS.avi',
        #                       cv2.VideoWriter_fourcc(*'MJPG'), 30, (height, width))
        path = '/home/shernandez/PycharmProjects/UMotorsport/v1/UnityTrainerPy-master/PyUMotorsport_v2/video_detections/'
        name = 'video_{:0>4d}.jpg'
        for i in range(len(video)):
            cv2.imwrite(path+name.format(i), video[i])
        # out.release()
        cv2.destroyAllWindows()
