from ObjectDetectionSegmentation.Networks.ModelManager import InferenceModel
from ObjectDetectionSegmentation.Data.DataManager import DataManager
import sys, time, os, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

'''
This script executes the network inference.
'''

if __name__ == '__main__':
    # Variables
    path_images = "/home/shernandez/PycharmProjects/UMotorsport/v1/Object-Detection-with-Segmentation-main/DetectionData/26_02_2021__16_59_0"
    path_images_destination = "/home/shernandez/PycharmProjects/UMotorsport/v1/Object-Detection-with-Segmentation-main/DetectionData/images_dest"
    path_newlabels = r'C:\Users\TTe_J\Downloads\new_labels.json'
    limit = 1 # <============================= unlabeled image limit
    model = "SNet_3L" # <============ models = HelperNetV1, SNet_5L0, SNet_4L, SNet_3L, SNet_3Lite
    output_type = "cls"  # regression = reg, classification = cls, regression + classficiation = reg+cls
    inference_type = "mask4seg"  # bbox4reg, bbox4seg, mask4reg or mask4seg
    min_area = 3  # <= for bbox4reg
    neighbours = 3  # <= for bbox4reg
    start_epoch = 100 # <============== trained epochs
    color_space = 82 # <====== bgr=None, lab=44, yuv=82, hsv=40, hsl=52
    specific_weights = "synthetic_real_cls_yuv"
    weights_path = f'Weights/{model}/{specific_weights}_epoch'
    labels = ["b", "y", "o_s", "o_b"]
    labels_color = {"b":(255, 102, 0), "y":(0, 254, 251), "o_s":(0, 186, 255), "o_b":(0, 133, 182)}
    # input_dims = (1, 720, 1280, 3)
    # input_dims = (1, 513, 1025, 3)
    # input_dims = (1, 180, 320, 3)
    input_dims = (1, 360, 640, 3)
    original_size = (720, 1280, 4)  # <= for bbox4reg

    # # Load the model and weigths
    # im_mask = InferenceModel(model, input_dims, weights_path, start_epoch, output_type, inference_type, original_size,
    #                     min_area, neighbours)
    # im_bbox = InferenceModel(model, input_dims, weights_path, start_epoch, output_type, "bbox4seg", original_size,
    #                          min_area, neighbours)
    #
    # # Overwrite control
    # if os.path.exists(path_newlabels):
    #     over = input(f"WARNING!!! Existing labels will be overwritten (overwrite or stop: o/s) => ")
    #     if over != 'o':
    #         print("Stoping")
    #         sys.exit()
    # if os.path.exists(path_images_destination):
    #     shutil.rmtree(path_images_destination)
    # os.makedirs(path_images_destination)


    # Load unlabeled images
    unlabed, names, sizes = DataManager.loadUnlabeled(path_images, path_images_destination, limit, color_space, input_dims[1:3])

    # Predict de una imagen en concreto
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    os.environ["CUDA_DEVICE_ORDER"] = '0'
    from tensorflow.python.saved_model import tag_constants
    saved_model_loaded = tf.saved_model.load("/home/shernandez/PycharmProjects/UMotorsport/v1/Object-Detection-with-Segmentation-main/Models/Net_7_saved_model", tags=[tag_constants.SERVING])
    model = saved_model_loaded.signatures['serving_default']
    for i in range(10):
        # y_hat = model(tf.constant(unlabed, dtype=tf.uint8))
        y_hat = model(tf.constant(unlabed))
    for i in range(100):
        times = []
        start = time.time()
        y_hat = model(tf.constant(unlabed))
        # y_hat = model(tf.constant(unlabed, dtype=tf.uint8))
        times.append(time.time()-start)
    print("time:", np.mean(times))
    # cap = cv2.VideoCapture(r'C:\Users\TTe_J\Downloads\ADCone.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter(r"C:\Users\TTe_J\Downloads\Imagenes\ADCone.mp4", fourcc, 30.0, (1280, 720))
    # idx = 0
    # while (cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret == False:
    #         break
    #     unlabed = np.expand_dims(cv2.cvtColor(cv2.resize(frame, (320, 180)), 82).astype('float32')/255., axis=0)
    #     y_hat = im_mask.predict(unlabed)
    #     labels_hat = im_bbox.predict(unlabed)
    #     masks = np.round(y_hat).astype(np.uint8)
    #     for i in range(len(masks)):
    #         mask = np.argmax(masks[i], axis=2)
    #         mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
    #         for m in range(masks[i].shape[2] - 1):
    #             contours, _ = cv2.findContours(np.uint8((mask == m) * 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             cv2.drawContours(frame, contours, -1, labels_color[labels[m]], 2)
    #         for y in labels_hat[i]:
    #             cv2.rectangle(frame, tuple(y[1]), tuple(y[2]), labels_color[labels[y[0]]], 4)
    #     out.write(frame)
    #     print(idx)
    #     idx += 1
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter(r"C:\Users\TTe_J\Downloads\Imagenes\UMotorsport.mp4", fourcc, 30.0, (1280, 720))
    # idx = 15997
    # while idx < 16687:
    #     frame = cv2.imread(r"C:\Users\TTe_J\Downloads\skidpad_v0.0"+f"/{idx}.png")
    #     unlabed = np.expand_dims(cv2.cvtColor(cv2.resize(frame, (320, 180)), 82).astype('float32') / 255., axis=0)
    #     y_hat = im_mask.predict(unlabed)
    #     labels_hat = im_bbox.predict(unlabed)
    #     masks = np.round(y_hat).astype(np.uint8)
    #     for i in range(len(masks)):
    #         mask = np.argmax(masks[i], axis=2)
    #         mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
    #         for m in range(masks[i].shape[2] - 1):
    #             contours, _ = cv2.findContours(np.uint8((mask == m) * 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             cv2.drawContours(frame, contours, -1, labels_color[labels[m]], 2)
    #         for y in labels_hat[i]:
    #             cv2.rectangle(frame, tuple(y[1]), tuple(y[2]), labels_color[labels[y[0]]], 4)
    #     out.write(frame)
    #     print(idx)
    #     idx += 1
    # out.release()
    # cv2.destroyAllWindows()

    # Save as json and show names of the modified files
    # vgg = DataManager.mask2vgg(np.round(y_hat).astype(np.uint8), labels, names, sizes, save_path=path_newlabels)
    # print(names)