import cv2
import numpy as np
from connection_utils.my_client import ConnectionManager
from controller_agent.agent import Agent
from cone_detection.detection_utils_v1 import ConeDetector
from trayectory_estimation.cone_processing import ConeProcessing
from cone_detection.detection_utils import BLUE_COLOR, YELLOW_COLOR, ORANGE_COLOR, DARK_ORANGE_COLOR
import utils
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

# Estas tres lineas resuelven algunos problemas con cuDNN en TF2 por los que no me permitía ejecutar en GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

def draw_plot(blue_center, yell_center, oran_left_center, oran_rigth_center, xmin, xmax, ymin, ymax, n_img=0):
    # Fuerzo el uso de backend TkAgg por que al instalar la API de detección de TF usa agg
    matplotlib.use("TkAgg")
    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)

    if len(blue_center) > 0:
        x = np.array(blue_center)[:, 0]
        y = image.shape[1] - np.array(blue_center)[:, 1]
        ax.plot(x, y, 'ob--')

    if len(yell_center) > 0:
        x = np.array(yell_center)[:, 0]
        y = image.shape[1] - np.array(yell_center)[:, 1]
        ax.plot(x, y, 'oy--')
    if len(oran_left_center) > 0:
        x = np.array(oran_left_center)[:, 0]
        y = image.shape[1] - np.array(oran_left_center)[:, 1]
        ax.plot(x, y, 'or--')
    if len(oran_rigth_center) > 0:
        x = np.array(oran_rigth_center)[:, 0]
        y = image.shape[1] - np.array(oran_rigth_center)[:, 1]
        ax.plot(x, y, 'om--')
    xlim = (xmin, xmax)
    ylim = (ymin, ymax)
    ax.set(xlim=xlim, ylim=ylim)
    plt.draw()
    plt.pause(10e-50)

    # Guardar video
    fig.savefig('/home/shernandez/PycharmProjects/UMotorsport/plt_images/plt_image{:03}.png'.format(n_img), bbox_inches='tight')

    # plt.show()
    # a = matplotlib.get_backend()

if __name__ == '__main__':
    checkpoint_path = '../cone_detection/saved_models/ResNet50_640x640_synt_2'
    detector = ConeDetector(checkpoint_path)
    cone_processing = ConeProcessing()
    agent = Agent()
    val = agent.valTrackbarsPID()
    # Inicializar conexiones con simulador
    connect_mng = ConnectionManager()
    # Guardar video
    out_image = cv2.VideoWriter('/home/shernandez/PycharmProjects/UMotorsport/image.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (1033, 579))

    verbose = 3  # 0 = no output, 1 = only telemetry, 2 = all message and paint images, verbose = 3 cone detection

    # Guardar video
    n_img = 0

    try:
        while True:

            # Pedir datos al simulador
            image, speed, throttle, steer, brake = connect_mng.get_data(verbose=1)
            if verbose == 3:
                # Detectar conos
                detections, [cone_centers, image, eagle_img] = detector.detect_cones(np.array(image), show_detections=False, min_score_thresh=0.3, real_time=True,
                                 im_name='output')

                # Transformo la imagen en vista de aguila
                src_trapezoide = utils.valTrackbars()  # Trapezoide a coger de la imagen original
                imgWarp_im = make_eagle_view(src_trapezoide)

                [warp_blue_center, warp_yell_center, warp_oran_center], \
                [order_warp_blue_center, order_warp_yell_center, order_warp_oran_left_center, order_warp_oran_rigth_center], \
                center = cone_processing.create_cone_map(detections, cone_centers, [eagle_img, image.shape])

                # Dibujo el centro, objetivo a alcanzar por el coche, en la imagen original
                image = cv2.circle(image, (center, 150), radius=2, color=(0, 255, 0), thickness=2)

                # Creo la imagen de coordenadas y pinto los conos
                coord_img = np.zeros((image.shape[1], image.shape[1], image.shape[2]), dtype=np.uint8)

                centro_img = int(coord_img.shape[0]/2)
                coord_img = draw_coord([(center, 100)], coord_img, (0, 255, 0))
                coord_img = draw_coord([(centro_img, 500)], coord_img, (255, 255, 255))
                coord_img = draw_coord(warp_blue_center, coord_img, BLUE_COLOR)
                coord_img = draw_coord(warp_yell_center, coord_img, YELLOW_COLOR)
                coord_img = draw_coord(order_warp_oran_rigth_center, coord_img, ORANGE_COLOR)
                coord_img = draw_coord(order_warp_oran_left_center, coord_img, DARK_ORANGE_COLOR)

                # Pinto las coordenadas ordenadas en matplotlib
                draw_plot(order_warp_blue_center, order_warp_yell_center, order_warp_oran_left_center, order_warp_oran_rigth_center, xmin=0, xmax=image.shape[1],
                          ymin=-200, ymax=image.shape[1], n_img=n_img)

                cv2.imshow("coord image", coord_img)
                cv2.imshow("eagle eye", imgWarp_im)
                cv2.imshow("detect", image)

                # Guardar video
                out_image.write(image)
                cv2.imwrite('/home/shernandez/PycharmProjects/UMotorsport/images/image{:03}.png'.format(n_img), image)
                n_img += 1

                cv2.waitKey(1)

                actions = agent.get_action(center, centro_img)

                connect_mng.send_actions(throttle=actions[0], brake=actions[1], steer=actions[2])

            if cv2.waitKey(1) == ord('q'):
                break

        connect_mng.close_connection()
    finally:
        # Guardar video
        out_image.release()

        connect_mng.close_connection()
