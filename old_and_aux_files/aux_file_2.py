import cv2
import numpy as np
from connection_utils.my_client import ConnectionManager
from controller_agent import agent
from cone_detection.detection_utils import ConeDetector
from cone_detection.detection_utils import BLUE_COLOR, YELLOW_COLOR, ORANGE_COLOR, DARK_ORANGE_COLOR
import utils
import matplotlib.pyplot as plt
import matplotlib
from simple_pid import PID
import tensorflow as tf
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Estas tres lineas resuelven algunos problemas con cuDNN en TF2 por los que no me permitía ejecutar en GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
Main script, where messages are received from the server and agent actions are sent.
'''
HSVvalues = [[73, 67, 27], [129, 255, 255], [29, 27, 57], [46, 255, 255], [5, 46, 32], [14, 180, 204]]
# intialTracbarVals = [[73, 67, 27], [129, 255, 255], [21, 58, 142], [24, 255, 255], [45, 28, 0, 70]]
intialTracbarVals = [[73, 67, 27], [129, 255, 255], [21, 58, 142], [24, 255, 255], [45, 27, 0, 44]]

utils.initializeTrackbars(intialTracbarVals[4])
utils.initializeTrackbarsPID([4, 7, 15])

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

def draw_ransac(blue_center, yell_center, oran_left_center, oran_rigth_center, xmin, xmax, ymin, ymax):
    # Fuerzo el uso de backend TkAgg por que al instalar la API de detección de TF usa agg
    matplotlib.use("TkAgg")
    plt.clf()
    fig = plt.figure(2)
    ax = fig.add_subplot(1, 1, 1)

    # Robustly fit linear model with RANSAC algorithm
    ransac = RANSACRegressor()

    if len(blue_center) > 0:
        x = np.array(blue_center)[:, 0]
        y = image.shape[1] - np.array(blue_center)[:, 1]

        ransac.fit(np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1)))
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_X = np.arange(x.min(), x.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)
        ax.plot(x[inlier_mask], y[inlier_mask], color='blue', marker='o', linestyle='None')
        ax.plot(x[outlier_mask], y[outlier_mask], color='cornflowerblue', marker='o', linestyle='None')
        plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=2,
                 label='RANSAC regressor')
    if len(yell_center) > 0:
        x = np.array(yell_center)[:, 0]
        y = image.shape[1] - np.array(yell_center)[:, 1]

        ransac.fit(np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1)))
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_X = np.arange(x.min(), x.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)
        ax.plot(x[inlier_mask], y[inlier_mask], color='yellow', marker='o', linestyle='None')
        ax.plot(x[outlier_mask], y[outlier_mask], color='yellowgreen', marker='o', linestyle='None')
        plt.plot(line_X, line_y_ransac, color='yellow', linewidth=2,
                 label='RANSAC regressor')
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
    # plt.show()
    # a = matplotlib.get_backend()

def draw_poly_ransac(blue_center, yell_center, oran_left_center, oran_rigth_center, xmin, xmax, ymin, ymax):
    # Fuerzo el uso de backend TkAgg por que al instalar la API de detección de TF usa agg
    matplotlib.use("TkAgg")
    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)

    ransac = make_pipeline(PolynomialFeatures(3), RANSACRegressor(random_state=42))

    if len(blue_center) > 0:
        x = np.array(blue_center)[:, 0]
        y = image.shape[1] - np.array(blue_center)[:, 1]
        ax.plot(x, y, 'ob--')
        x_plot = np.linspace(x.min(), x.max())
        ransac.fit(np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1)))
        y_plot = ransac.predict(x_plot[:, np.newaxis])
        plt.plot(x_plot, y_plot, color='blue', linestyle='o', linewidth=2)

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
    # plt.show()
    # a = matplotlib.get_backend()

if __name__ == '__main__':
    checkpoint_path = '../cone_detection/saved_models/ResNet50_640x640_synt_2'
    detector = ConeDetector(checkpoint_path)

    # Inicializar conexiones con simulador
    connect_mng = ConnectionManager()
    # Guardar video
    out_image = cv2.VideoWriter('/home/shernandez/PycharmProjects/UMotorsport/image_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (1033, 579))
    cap = cv2.VideoCapture('chaplin.mp4')

    verbose = 3  # 0 = no output, 1 = only telemetry, 2 = all message and paint images, verbose = 3 cone detection

    # Guardar video
    n_img = 0
    try:
        while cap.isOpened():

            # Pedir datos al simulador
            # image, speed, throttle, steer, brake = connect_mng.get_data(verbose=1)
            speed = throttle = steer = brake = 0.
            _, image = cap.read()

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
                # list_blue_center, list_yell_center, list_oran_left, list_oran_rigth = detector.join_cones(cone_centers)
                list_blue_center, list_yell_center, list_oran_center = detector.split_cones(cone_centers)

                # # coger el centro entre el tercer cono azul y amarillo
                # if len(list_yell_center) > 3 and len(list_blue_center) > 3:
                #     centro = np.int32(((list_yell_center[2] - list_blue_center[2])/2) + list_blue_center[2])
                #     image = cv2.circle(image, (centro[0], centro[1]), radius=2, color=(0, 255, 0), thickness=2)
                #
                # # pinto las uniones entre conos
                # image, eagle_img = draw_join_cones(image, eagle_img, list_blue_center, BLUE_COLOR)
                # image, eagle_img = draw_join_cones(image, eagle_img, list_yell_center, YELLOW_COLOR)
                # image, eagle_img = draw_join_cones(image, eagle_img, list_oran_left, ORANGE_COLOR)
                # image, eagle_img = draw_join_cones(image, eagle_img, list_oran_rigth, DARK_ORANGE_COLOR)

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
                # warp_oran_rigth_center = utils.perspective_warp_coordinates(list_oran_rigth,
                #                                                             eagle_img,
                #                                                             dst_size=(image.shape[1], image.shape[1]),
                #                                                             src=src_trapezoide)
                # warp_oran_left_center = utils.perspective_warp_coordinates(list_oran_left,
                #                                                            eagle_img,
                #                                                            dst_size=(
                #                                                                image.shape[1], image.shape[1]),
                #                                                            src=src_trapezoide)
                warp_oran_center = utils.perspective_warp_coordinates(list_oran_center,
                                                                      eagle_img,
                                                                      dst_size=(
                                                                           image.shape[1], image.shape[1]),
                                                                      src=src_trapezoide)


                # algoritmo para unir conos contiguos. Lo aplicamos sobre los conos en perspectiva
                order_warp_blue_center = detector.join_cones(warp_blue_center, unique_color='blue')
                order_warp_yell_center = detector.join_cones(warp_yell_center, unique_color='yell')
                order_warp_oran_left_center, order_warp_oran_rigth_center = detector.join_cones(warp_oran_center, unique_color='oran')

                # coger el centro entre el tercer cono azul y amarillo
                # if len(list_yell_center) > 3 and len(list_blue_center) > 3:
                #     centro = np.int32(((list_yell_center[2] - list_blue_center[2]) / 2) + list_blue_center[2])
                #     image = cv2.circle(image, (centro[0], centro[1]), radius=2, color=(0, 255, 0), thickness=2)
                # x1 = stats.mode(np.array(warp_blue_center)[:, 0]).mode[0]
                # x2 = stats.mode(np.array(warp_yell_center)[:, 0]).mode[0]
                if len(warp_blue_center) > 1 and len(warp_yell_center) > 1:
                    x1 = np.median(np.array(warp_blue_center)[:, 0])
                    x2 = np.median(np.array(warp_yell_center)[:, 0])
                    c = int((x2 - x1) / 2 + x1)
                elif len(order_warp_oran_left_center) > 1 and len(order_warp_oran_rigth_center) > 1:
                    x1 = np.median(np.array(order_warp_oran_left_center)[:, 0])
                    x2 = np.median(np.array(order_warp_oran_rigth_center)[:, 0])
                    c = int((x2 - x1) / 2 + x1)
                else:
                    c = 0.

                # Dibujo el centro, objetivo a alcanzar por el coche, en la imagen original
                image = cv2.circle(image, (c, 150), radius=2, color=(0, 255, 0), thickness=2)

                # Creo la imagen de coordenadas y pinto los conos
                coord_img = np.zeros((image.shape[1], image.shape[1], image.shape[2]), dtype=np.uint8)

                centro_img = int(coord_img.shape[0]/2)
                coord_img = draw_coord([(c, 100)], coord_img, (0, 255, 0))
                coord_img = draw_coord([(centro_img, 500)], coord_img, (255, 255, 255))
                coord_img = draw_coord(warp_blue_center, coord_img, BLUE_COLOR)
                coord_img = draw_coord(warp_yell_center, coord_img, YELLOW_COLOR)
                coord_img = draw_coord(order_warp_oran_rigth_center, coord_img, ORANGE_COLOR)
                coord_img = draw_coord(order_warp_oran_left_center, coord_img, DARK_ORANGE_COLOR)

                ref_point = centro_img - c
                val = utils.valTrackbarsPID()
                pid = PID(Kp=val[0], Ki=val[1], Kd=val[2], setpoint=0, output_limits=(-1., 1.))
                giro = pid(ref_point)

                # Pinto las coordenadas ordenadas en matplotlib
                draw_plot(order_warp_blue_center, order_warp_yell_center, order_warp_oran_left_center, order_warp_oran_rigth_center, xmin=0, xmax=image.shape[1],
                          ymin=-200, ymax=image.shape[1], n_img=n_img)

                # Pinto las coordenadas y la regresión por ransac en matplotlib
                # draw_ransac(warp_blue_center, warp_yell_center, order_warp_oran_left_center,
                #           order_warp_oran_rigth_center, xmin=0, xmax=image.shape[1],
                #           ymin=-200, ymax=image.shape[1])

                cv2.imshow("coord image", coord_img)
                cv2.imshow("eagle eye", imgWarp_im)
                cv2.imshow("detect", image)

                # Guardar video
                out_image.write(image)
                cv2.imwrite('/home/shernandez/PycharmProjects/UMotorsport/images/image{:03}.png'.format(n_img), image)
                n_img += 1

                cv2.waitKey(1)

            actions = agent.testAction(image, speed=0.0, throttle=0.1, steer=giro, brake=0.0)

            connect_mng.send_actions(throttle=actions[0], steer=actions[1], brake=actions[2])

            if cv2.waitKey(1) == ord('q'):
                break

        # connect_mng.close_connection()
    finally:
        # Guardar video
        out_image.release()

        # connect_mng.close_connection()
