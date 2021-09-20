import time, cv2
import numpy as np
from PIL import Image
from myTools.my_client import bind2server
from myTools.my_client import ConnectionManager
import agent
from cone_detection.detection_utils import ConeDetector
from cone_detection.detection_utils import BLUE_COLOR, YELLOW_COLOR, ORANGE_COLOR, DARK_ORANGE_COLOR
import utils
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy import stats
from simple_pid import PID

"""
esto es una prueba
"""
'''
Main script, where messages are received from the server and agent actions are sent.
'''
HSVvalues = [[73, 67, 27], [129, 255, 255], [29, 27, 57], [46, 255, 255], [5, 46, 32], [14, 180, 204]]
# intialTracbarVals = [[73, 67, 27], [129, 255, 255], [21, 58, 142], [24, 255, 255], [45, 28, 0, 70]]
intialTracbarVals = [[73, 67, 27], [129, 255, 255], [21, 58, 142], [24, 255, 255], [45, 27, 0, 44]]

utils.initializeTrackbars(intialTracbarVals[4])
utils.initializeTrackbarsPID([32, 7, 15])

def draw_centers(cone_centers, image, eagle_img):
    for center in cone_centers:
        c = center[0]
        color = center[1]
        if color == detector.blue_color:
            color = BLUE_COLOR
        elif color == detector.yellow_color:
            color = YELLOW_COLOR
        elif color == detector.orange_color:
            color = ORANGE_COLOR

        image = cv2.circle(image, c, radius=2, color=color, thickness=2)
        eagle_img = cv2.circle(eagle_img, c, radius=0, color=color, thickness=-1)

    return image, eagle_img

def draw_join_cones(image, eagle_img, center_list, color):
    for i in range(len(center_list) - 1):
        x1 = center_list[i][0]
        y1 = center_list[i][1]
        x2 = center_list[i + 1][0]
        y2 = center_list[i + 1][1]
        cv2.line(image, (x1, y1), (x2, y2), color, thickness=2)
        cv2.line(eagle_img, (x1, y1), (x2, y2), color, thickness=1)

    return image, eagle_img

def make_eagle_view(src_trapezoide):
    # pers_img = np.ones((image.shape[0], int(image.shape[1] * 2), image.shape[2]), dtype=np.uint8)
    # pers_img[0:image.shape[0], image.shape[1] // 2:(image.shape[1] // 2) + image.shape[1], :] = eagle_img
    # src_trapezoide = np.float32([(0.0, 0.0), (0.58, 0.65), (0.1, 1), (1, 1)])
    # src_trapezoide = utils.valTrackbars()
    imgWarp_im = utils.perspective_warp(eagle_img,
                                        dst_size=(image.shape[1], image.shape[1]),
                                        src=src_trapezoide)
    return imgWarp_im

def draw_coord(warp_center, coord_img, color):
    for coord in warp_center:
        if coord[0] >= 0 and coord[1] >= 0:
            coord_img = cv2.circle(coord_img, (coord[0], coord[1]), radius=2, color=color, thickness=3)
    return coord_img



def draw_plot(blue_center, yell_center, oran_left_center, oran_rigth_center, xmin, xmax, ymin, ymax):
    # Fuerzo el uso de backend TkAgg por que al instalar la API de detección de TF usa agg
    matplotlib.use("TkAgg")
    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)

    if len(blue_center) > 0:
        x = np.array(blue_center)[:, 0]
        y = image.shape[1] - np.array(blue_center)[:, 1]
        ax.plot(x, y, 'ob')
    if len(yell_center) > 0:
        x = np.array(yell_center)[:, 0]
        y = image.shape[1] - np.array(yell_center)[:, 1]
        ax.plot(x, y, 'oy')
    if len(oran_left_center) > 0:
        x = np.array(oran_left_center)[:, 0]
        y = image.shape[1] - np.array(oran_left_center)[:, 1]
        ax.plot(x, y, 'or')
    if len(oran_rigth_center) > 0:
        x = np.array(oran_rigth_center)[:, 0]
        y = image.shape[1] - np.array(oran_rigth_center)[:, 1]
        ax.plot(x, y, 'om')
    xlim = (xmin, xmax)
    ylim = (ymin, ymax)
    ax.set(xlim=xlim, ylim=ylim)
    plt.draw()
    plt.pause(10e-50)
    # plt.show()
    # a = matplotlib.get_backend()

if __name__ == '__main__':
    checkpoint_path = './cone_detection/saved_models/ResNet50_640x640_synt_2'
    detector = ConeDetector(checkpoint_path)

    # Inicializar conexiones con simulador
    connect_mng = ConnectionManager()

    verbose = 3  # 0 = no output, 1 = only telemetry, 2 = all message and paint images, verbose = 3 cone detection


    try:
        while True:

            # Pedir datos al simulador
            image, speed, throttle, steer, brake = connect_mng.get_data(verbose=1)
            if verbose == 3:
                # Detectar conos
                detections = detector.detect_in_image(np.array(image), plot=False, min_score_thresh=0.3, real_time=True,
                                 im_name='output')

                # Cojo rectángulos de las detecciones y filtro los conos por color
                rectangles = np.array(detections["detection_boxes"])[0]
                orig_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image, cone_centers, cone_bases = detector.color_filter_cones(image, rectangles, paint_rectangles=True, bgr=False)

                # Creo la imagen de vista de águila
                eagle_img = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8) + np.uint8(orig_image*0.5)

                # Pinto los centros de los conos en las imágenes
                image, eagle_img = draw_centers(cone_centers, image, eagle_img)

                # algoritmo para unir conos contiguos
                list_blue_center, list_yell_center, list_oran_left, list_oran_rigth = detector.join_cones(cone_centers)

                # coger el centro entre el tercer cono azul y amarillo
                if len(list_yell_center) > 3 and len(list_blue_center) > 3:
                    centro = np.int32(((list_yell_center[2] - list_blue_center[2])/2) + list_blue_center[2])
                    image = cv2.circle(image, (centro[0], centro[1]), radius=2, color=(0, 255, 0), thickness=2)

                # pinto las uniones entre conos
                image, eagle_img = draw_join_cones(image, eagle_img, list_blue_center, BLUE_COLOR)
                image, eagle_img = draw_join_cones(image, eagle_img, list_yell_center, YELLOW_COLOR)
                image, eagle_img = draw_join_cones(image, eagle_img, list_oran_left, ORANGE_COLOR)
                image, eagle_img = draw_join_cones(image, eagle_img, list_oran_rigth, DARK_ORANGE_COLOR)

                # Transformo la imagen en vista de aguila
                src_trapezoide = utils.valTrackbars()  # Trapezoide a coger de la imagen original
                imgWarp_im = make_eagle_view(src_trapezoide)

                # Transformo las coordenadas a vista de águila
                warp_blue_center = utils.perspective_warp_coordinates(list_blue_center,
                                                                      eagle_img,
                                                                      dst_size=(image.shape[1], image.shape[1]),
                                                                      src=src_trapezoide)
                warp_yell_center = utils.perspective_warp_coordinates(list_yell_center,
                                                                      eagle_img,
                                                                      dst_size=(image.shape[1], image.shape[1]),
                                                                      src=src_trapezoide)
                warp_oran_rigth_center = utils.perspective_warp_coordinates(list_oran_rigth,
                                                                            eagle_img,
                                                                            dst_size=(image.shape[1], image.shape[1]),
                                                                            src=src_trapezoide)
                warp_oran_left_center = utils.perspective_warp_coordinates(list_oran_left,
                                                                           eagle_img,
                                                                           dst_size=(
                                                                               image.shape[1], image.shape[1]),
                                                                           src=src_trapezoide)

                # coger el centro entre el tercer cono azul y amarillo
                if len(list_yell_center) > 3 and len(list_blue_center) > 3:
                    centro = np.int32(((list_yell_center[2] - list_blue_center[2]) / 2) + list_blue_center[2])
                    image = cv2.circle(image, (centro[0], centro[1]), radius=2, color=(0, 255, 0), thickness=2)
                # x1 = stats.mode(np.array(warp_blue_center)[:, 0]).mode[0]
                # x2 = stats.mode(np.array(warp_yell_center)[:, 0]).mode[0]
                x1 = np.median(np.array(warp_blue_center)[:, 0])
                x2 = np.median(np.array(warp_yell_center)[:, 0])
                c = int(( x2-x1)/2 + x1)
                # Creo la imagen de coordenadas y pinto los conos
                coord_img = np.zeros((image.shape[1], image.shape[1], image.shape[2]), dtype=np.uint8)

                centro_img = int(coord_img.shape[0]/2)
                coord_img = draw_coord([(c, 100)], coord_img, (0, 255, 0))
                coord_img = draw_coord([(centro_img, 500)], coord_img, (255, 255, 255))
                coord_img = draw_coord(warp_blue_center, coord_img, BLUE_COLOR)
                coord_img = draw_coord(warp_yell_center, coord_img, YELLOW_COLOR)
                coord_img = draw_coord(warp_oran_rigth_center, coord_img, ORANGE_COLOR)
                coord_img = draw_coord(warp_oran_left_center, coord_img, DARK_ORANGE_COLOR)

                ref_point = centro_img - c
                val = utils.valTrackbarsPID()
                pid = PID(Kp=val[0], Ki=val[1], Kd=val[2], setpoint=0, output_limits=(-1., 1.))
                giro = pid(ref_point)

                # Pinto las coordenadas en matplotlib
                draw_plot(warp_blue_center, warp_yell_center, warp_oran_left_center, warp_oran_rigth_center, xmin=0, xmax=image.shape[1],
                          ymin=-200, ymax=image.shape[1])

                cv2.imshow("coord image", coord_img)
                cv2.imshow("eagle eye", imgWarp_im)
                cv2.imshow("detect", image)
                cv2.waitKey(1)

            actions = agent.testAction(image, speed=0.0, throttle=0.01, steer=giro, brake=0.0)

            connect_mng.send_actions(throttle=actions[0], steer=actions[1], brake=actions[2])

            if cv2.waitKey(1) == ord('q'):
                break

        connect_mng.close_connection()
    finally:
        connect_mng.close_connection()
