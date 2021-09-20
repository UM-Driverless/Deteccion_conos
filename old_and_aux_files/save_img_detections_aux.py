import time, os, cv2
import numpy as np
from PIL import Image
from connection_utils.my_client import bind2server
from cone_detection.detection_utils import ConeDetector


if __name__ == '__main__':
    checkpoint_path = '../cone_detection/saved_models/ResNet50_640x640_synt_2'
    detector = ConeDetector(checkpoint_path)
    # Image processing
    image = cv2.imread("/home/shernandez/PycharmProjects/UMotorsport/Deteccion_conos/cliente_simulador/PyUMotorsport/imagenes_auxiliares/imagen_conos.png")
    w = image.shape[1]
    h = image.shape[0]
    detections = detector.detect_in_image(np.array(image), plot=False, min_score_thresh=0.1, real_time=False, im_name='Left')
    rectangles = np.array(detections["detection_boxes"])[0]

    for i in range(15):
        ymin = int(rectangles[i][0] * h)-2
        xmin = int(rectangles[i][1] * w)-2
        ymax = int(rectangles[i][2] * h)+2
        xmax = int(rectangles[i][3] * w)+2
        rect_im = image[ymin:ymax, xmin:xmax, :]
        cv2.imshow("cone", rect_im)
        # cv2.imwrite("imagenes_auxiliares/cone_"+str(i)+".png", cv2.cvtColor(rect_im, cv2.COLOR_RGB2BGR))
    cv2.rectangle(image, (xmax, ymax), (xmin, ymin), (0, 255, 0), 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("detections", image)

    cv2.waitKey()
    # detector.detect_in_image(np.array(image[1]), plot=True, min_score_thresh=0.3, real_time=True, im_name='Rigth')


