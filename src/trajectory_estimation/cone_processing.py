import numpy as np
import cv2

class ConeProcessing():
    def __init__(self):

        intialTracbarVals = [[73, 67, 27], [129, 255, 255], [21, 58, 142], [24, 255, 255], [45, 27, 0, 44]]
        self.initializeTrackbars(intialTracbarVals[4])

    def create_cone_map2(self, cone_centers):
        '''
        In progress
        
        Takes cone_centers, 
        '''
        pass

    def create_cone_map(self, cones, aux_data=None, orig_im_shape=(1, 180, 320, 3), img_to_wrap=None):
        """
        
        Returns data array, which ...
        data[0] = cone centers per color in top view = [warp_blue_center, warp_yell_center, warp_oran_center]
        data[1] = cone centers per size (left and right) [{list with centers of blue cones}, [= for yellow cones], ...] list with centers of each cone
        data[2] = reference point to see what's left and right sides
        
        
        Performs the cones detection task. The detection must include the bounding boxes and classification of each
        cone.
        :param img: (3D numpy array) Image to process.
        :param min_score_thresh: (float in [0., 1.]) Min score of confident on a detection.
        :param show_detections: (bool) If True: The image with detections id displayed. If False: no image is displayed.
        :param im_name: (string) Name of the detection image when show_detections=True
        :return: [ndarray, list] ndarray with detected bounding boxes and classification of each cone, list of auxiliar
                                data.
        """
        
        cone_centers = [
            [cone['coords'] for cone in cones if cone['label'] == 'blue_cone'],
            [cone['coords'] for cone in cones if cone['label'] == 'yellow_cone'],
            [cone['coords'] for cone in cones if cone['label'] == 'orange_cone'],
            [cone['coords'] for cone in cones if cone['label'] == 'large_orange_cone'],
            [cone['coords'] for cone in cones if cone['label'] == 'unknown_cone'],
            [cone['coords'] for cone in cones if cone['label'] == 'unknown_cone']
        ]
        
        src_trapezoide = self.valTrackbars()  # Trapezoide a coger de la imagen original

        if np.array(cone_centers[0]).shape[0]:
            list_blue_center = cone_centers[0]
            # Transformo las coordenadas a vista de águila
            warp_blue_center = self.perspective_warp_coordinates(list_blue_center,
                                                                 orig_im_shape,
                                                                 dst_size=(orig_im_shape[2], orig_im_shape[2]),
                                                                 src=src_trapezoide)
        else:
            list_blue_center = []
            warp_blue_center = []

        if np.array(cone_centers[1]).shape[0]:
            list_yell_center = cone_centers[1]
            warp_yell_center = self.perspective_warp_coordinates(list_yell_center,
                                                                 orig_im_shape,
                                                                 dst_size=(orig_im_shape[2], orig_im_shape[2]),
                                                                 src=src_trapezoide)
        else:
            list_yell_center = []
            warp_yell_center = []

        if np.array(cone_centers[2]).shape[0]:
            list_oran_center = cone_centers[2]
            if np.array(cone_centers[3]).shape[0]:
                list_oran_center = np.concatenate([list_oran_center, cone_centers[3]])
            warp_oran_center = self.perspective_warp_coordinates(list_oran_center,
                                                                 orig_im_shape,
                                                                 dst_size=(orig_im_shape[2], orig_im_shape[2]),
                                                                 src=src_trapezoide)
        elif np.array(cone_centers[3]).shape[0]:
            list_oran_center = cone_centers[3]
            warp_oran_center = self.perspective_warp_coordinates(list_oran_center,
                                                                 orig_im_shape,
                                                                 dst_size=(orig_im_shape[2], orig_im_shape[2]),
                                                                 src=src_trapezoide)
        else:
            list_oran_center = []
            warp_oran_center = []

        order_warp_blue_center = warp_blue_center
        order_warp_yell_center = warp_yell_center
        order_warp_oran_left_center, order_warp_oran_rigth_center = self.split_orange_cones(warp_oran_center)

        # calcular el centro.
        # Target point will be the average of 2 points, only taking the horizontal coordinate into account:
        # The left one will be the median x coordinate of blue cones
        # The right one will be the median x coordinate of yellow cones
        
        if len(warp_blue_center) > 1 and len(warp_yell_center) > 1:
            x1 = np.median(np.array(warp_blue_center)[:, 0])
            x2 = np.median(np.array(warp_yell_center)[:, 0])
            center = int((x1 + x2) / 2)
        elif len(order_warp_oran_left_center) > 1 and len(order_warp_oran_rigth_center) > 1:
            x1 = np.median(np.array(order_warp_oran_left_center)[:, 0])
            x2 = np.median(np.array(order_warp_oran_rigth_center)[:, 0])
            center = int((x1 + x2) / 2)
        else:
            center = 0.

        return [warp_blue_center, warp_yell_center, warp_oran_center], \
               [np.array(order_warp_blue_center), np.array(order_warp_yell_center),
                np.array(order_warp_oran_left_center), np.array(order_warp_oran_rigth_center)], \
                center

    def initializeTrackbars(self, intialTracbarVals):
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 360, 240)
        cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0], 50, self.nothing)
        cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], 100, self.nothing)
        cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], 50, self.nothing)
        cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], 100, self.nothing)


    def valTrackbars(self):
        widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
        heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
        widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
        heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")

        src = np.float32([(widthTop / 100, heightTop / 100), (1 - (widthTop / 100), heightTop / 100),
                          (widthBottom / 100, heightBottom / 100), (1 - (widthBottom / 100), heightBottom / 100)])
        # src = np.float32([[0.45, 0.27], [0.55, 0.27], [0., 0.44], [1, 0.44]])
        return src

    def nothing(self, x):
        pass

    def split_orange_cones(self, cone_centers):
        """
        :cone_centers list: [[(x_1, y_1), color_tag_1], ..., [(x_n, y_n), color_tag_n]]
        :unique_color string: 'blue'
                              'yell'
                              'oran'
                              Si se selecciona un único color para la lista de conos la entrada esperada para
                              cone_centers será: [[x_1, y_1], ... , [x_n, y_n]]
        """
        if len(cone_centers) > 0:

            median = np.median(cone_centers[:, 0])


            list_oran_left = []
            list_oran_right = []
            for c in cone_centers:
                if c[0] > median:
                    list_oran_right.append(c)
                else:
                    list_oran_left.append(c)
            return np.array(list_oran_left), np.array(list_oran_right)
        return np.array([]), np.array([])

    def take_two_first_oranges(self, cone_list):
        index = np.argmax(cone_list[:, 1])
        first_cone = cone_list[index]
        cone_list = np.delete(cone_list, index, axis=0)
        four_first_cones = []
        for i in range(np.minimum(4, cone_list.shape[0])):
            index = np.argmax(cone_list[:, 1])
            four_first_cones.append(cone_list[index])
            cone_list = np.delete(cone_list, index, axis=0)

        # distances = self.calc_distances(first_cone, four_first_cones)
        x_distances = []
        y_distances = []
        for cone in four_first_cones:
            x_distances.append(np.abs(first_cone[0] - cone[0]))
            y_distances.append(np.abs(first_cone[1] - cone[1]))

        # coger los dos más lejanos en x
        lejano_1 = np.argmax(x_distances)
        x_distances[lejano_1] = 0
        lejano_2 = np.argmax(x_distances)

        if y_distances[lejano_1] < y_distances[lejano_2]:
            second_cone = four_first_cones[lejano_1]
        else:
            second_cone = four_first_cones[lejano_2]

        if first_cone[0] < second_cone[0]:
            left = first_cone
            rigth = second_cone
        else:
            rigth = first_cone
            left = second_cone
        return left, rigth

    def calc_distances(self, point, point_list):
        distances = [np.linalg.norm(point - p) for p in point_list]
        return distances

    def _join_check_cones(self, b_center, list_blue_center):
        # Cogemos el cono que más abajo esté en la imagen
        index_b = np.argmax(b_center[:, 1])
        last_point = b_center[index_b]
        list_blue_center.append(last_point)
        b_center = np.delete(b_center, index_b, axis=0)
        iter = 0
        while len(b_center) > 0:
            # Calculamos distancia a otros conos y cogemos el más cercano
            try:
                angle_condition = False
                b_center_copy = b_center
                while not angle_condition and len(b_center_copy) > 0:

                    distances = self.calc_distances(last_point, b_center_copy)
                    index_b = np.argmin(distances)
                    # check for angle
                    if len(list_blue_center) > 0:
                        # p2 = list_blue_center[iter-1]
                        # p3 = b_center[index_b]
                        if len(list_blue_center) > 1:
                            # p1 = list_blue_center[iter - 2]
                            v1 = list_blue_center[iter - 1] - list_blue_center[iter - 2]
                        else:
                            # TODO: revisar por que no sale el vector en la dirección correcta
                            # Calculando el vector entre (v_x, v_y) y (v_x, v_y-10). Esto se hace para suponer
                            # que la dirección del vector desde el punto inicial es vertical
                            # p1 = [list_blue_center[iter - 1][0], list_blue_center[iter - 1][1]+10]
                            v1 = list_blue_center[iter - 1] - [list_blue_center[iter - 1][0],
                                                               list_blue_center[iter - 1][1] + 10]
                        # plt.clf()
                        # fig = plt.figure(5)
                        # ax = fig.add_subplot(1, 1, 1)
                        # max = 714 # np.max(np.array(b_center)[:, 1])
                        # x = np.array(b_center)[:, 0]
                        # y = max - np.array(b_center)[:, 1]
                        # ax.plot(x, y, '^b')
                        #
                        # ax.plot(p1[0], max - p1[1], 'or')
                        # ax.plot(p2[0], max - p2[1], 'oy')
                        # ax.plot(p3[0], max - p3[1], 'om')
                        # ax.set(xlim=(0, 714), ylim=(-200, 714))
                        # plt.draw()
                        # plt.pause(10e-50)

                        v2 = b_center[index_b] - list_blue_center[iter - 1]
                        if v2[1] == 0. and v2[0] == 0.0:
                            # TODO: si esto ocurre, estoy evaluando ir al mismo punto ¿por que pasa esto?
                            pass
                        angle = self.angle_between(v1, v2)
                    else:
                        angle = 0.

                    radians_limit = np.pi / 4.  # np.deg2rad(45)
                    # radians_limit = 0.05
                    angle_condition = angle <= radians_limit
                    if angle_condition:
                        point_selected = b_center_copy[index_b]
                    else:
                        b_center_copy = np.delete(b_center_copy, index_b, axis=0)


            except:
                pass
            # last_point = b_center[index_b]
            if angle_condition:
                last_point = point_selected
                list_blue_center.append(last_point)
                b_center = np.delete(b_center, index_b, axis=0)
            iter += 1
            if len(b_center) < 1 or iter > 40:
                break

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        norm = np.linalg.norm(vector)
        if norm != 0.:
            u_vector = vector / norm
        else:
            u_vector = [0., 0.]
        return u_vector

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
                angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                angle_between((1, 0, 0), (1, 0, 0))
                0.0
                angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))