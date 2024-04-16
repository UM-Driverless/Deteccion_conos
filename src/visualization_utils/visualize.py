import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Visualize:
    def __init__(self):
        self.BLUE_COLOR = (255, 0, 0)
        self.YELLOW_COLOR = (0, 237, 255)
        self.ORANGE_COLOR = (0, 137, 255)
        self.DARK_ORANGE_COLOR = (0, 87, 255)

        self.n_img = 0

    def show_cone_map(self, ref_point, warp_blue_center, warp_yell_center,
                      order_warp_oran_left_center, order_warp_oran_rigth_center, image_shape):

        # Creo la imagen de coordenadas y pinto los conos
        cenital_img = np.zeros((image_shape[1], image_shape[1], image_shape[2]), dtype=np.uint8)

        # Sacar centro de la imagen
        img_center = int(cenital_img.shape[0] / 2)

        # Pintar posición de los conos en vista de aguila
        cenital_img = self.draw_coord([(ref_point, 100)], cenital_img, (0, 255, 0))
        cenital_img = self.draw_coord([(img_center, 500)], cenital_img, (255, 255, 255))
        cenital_img = self.draw_coord(warp_blue_center, cenital_img, self.BLUE_COLOR)
        cenital_img = self.draw_coord(warp_yell_center, cenital_img, self.YELLOW_COLOR)
        cenital_img = self.draw_coord(order_warp_oran_rigth_center, cenital_img, self.ORANGE_COLOR)
        cenital_img = self.draw_coord(order_warp_oran_left_center, cenital_img, self.DARK_ORANGE_COLOR)
        cv2.imshow("coord image", cenital_img)

    def draw_joined_cones(self, order_warp_blue_center, order_warp_yell_center, order_warp_oran_left_center,
                                 order_warp_oran_rigth_center, image_shape):
        # Pinto las coordenadas ordenadas en matplotlib
        self.draw_plot(order_warp_blue_center, order_warp_yell_center, order_warp_oran_left_center,
                                 order_warp_oran_rigth_center, xmin=0, xmax=image_shape[1],
                                 ymin=-200, ymax=image_shape[1])

    def show_detections(self, detections, image, detector):
        # Cojo rectángulos de las detecciones y filtro los conos por color
        rectangles = np.array(detections["detection_boxes"])[0]
        image, cone_centers, cone_bases = detector._color_filter_cones(image, rectangles, paint_rectangles=True,
                                                                   bgr=False)
        # Creo la imagen de vista de águila
        orig_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        eagle_img = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8) + np.uint8(
            orig_image * 0.5)

        # Pinto los centros de los conos en las imágenes
        image = detector._draw_centers(cone_centers, image)

        # Mostrar detecciones
        cv2.imshow("detect", image)

    def draw_coord(self, warp_center, cenital_img, color):
        for coord in warp_center:
            if coord[0] >= 0 and coord[1] >= 0:
                cenital_img = cv2.circle(cenital_img, (coord[0], coord[1]), radius=2, color=color, thickness=3)
        return cenital_img

    def draw_plot(self, blue_center, yell_center, oran_left_center, oran_rigth_center, xmin, xmax, ymin, ymax):
        # Fuerzo el uso de backend TkAgg por que al instalar la API de detección de TF usa agg
        matplotlib.use("TkAgg")
        plt.clf()
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)

        if len(blue_center) > 0:
            x = np.array(blue_center)[:, 0]
            y = xmax - np.array(blue_center)[:, 1]
            ax.plot(x, y, 'ob--')

        if len(yell_center) > 0:
            x = np.array(yell_center)[:, 0]
            y = xmax - np.array(yell_center)[:, 1]
            ax.plot(x, y, 'oy--')
        if len(oran_left_center) > 0:
            x = np.array(oran_left_center)[:, 0]
            y = xmax - np.array(oran_left_center)[:, 1]
            ax.plot(x, y, 'or--')
        if len(oran_rigth_center) > 0:
            x = np.array(oran_rigth_center)[:, 0]
            y = xmax - np.array(oran_rigth_center)[:, 1]
            ax.plot(x, y, 'om--')
        xlim = (xmin, xmax)
        ylim = (ymin, ymax)
        ax.set(xlim=xlim, ylim=ylim)
        plt.draw()
        plt.pause(10e-50)

        # Guardar video
        fig.savefig('/home/shernandez/PycharmProjects/UMotorsport/plt_images/plt_image{:03}.png'.format(self.n_img),
                    bbox_inches='tight')
        self.n_img += 1

        # plt.show()
        # a = matplotlib.get_backend()
