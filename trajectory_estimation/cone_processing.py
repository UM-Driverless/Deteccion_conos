from trajectory_estimation.cone_processing_interface import ConeProcessingInterface
import numpy as np
import cv2


class ConeProcessing(ConeProcessingInterface):
    def __init__(self):
        super().__init__()
        self.blue_color = 'blue'
        self.yellow_color = 'yell'
        self.orange_color = 'oran'

        intialTracbarVals = [[73, 67, 27], [129, 255, 255], [21, 58, 142], [24, 255, 255], [45, 27, 0, 44]]
        self.initializeTrackbars(intialTracbarVals[4])

    def create_cone_map(self, cone_centers, labels, aux_data=None, orig_im_shape=(1, 180, 320, 3), img_to_wrap=None):
        """
        
        Returns data array, which ...
        data[0] = cone centers per color in top view = [warp_blue_center, warp_yell_center, warp_oran_center]
        data[1] = [{list with centers of blue cones}, [= for yellow cones], ...] list with centers of each cone
        data[2] (=data[-2]) returns the reference point
        
        
        Performs the cones detection task. The detection must include the bounding boxes and classification of each
        cone.
        :param img: (3D numpy array) Image to process.
        :param min_score_thresh: (float in [0., 1.]) Min score of confident on a detection.
        :param show_detections: (bool) If True: The image with detections id displayed. If False: no image is displayed.
        :param im_name: (string) Name of the detection image when show_detections=True
        :return: [ndarray, list] ndarray with detected bounding boxes and classification of each cone, list of auxiliar
                                data.
        """
        src_trapezoide = self.valTrackbars()  # Trapezoide a coger de la imagen original

        if cone_centers[0].shape[0]:
            list_blue_center = cone_centers[0]
            # Transformo las coordenadas a vista de águila
            warp_blue_center = self.perspective_warp_coordinates(list_blue_center,
                                                                 orig_im_shape,
                                                                 dst_size=(orig_im_shape[2], orig_im_shape[2]),
                                                                 src=src_trapezoide)
        else:
            list_blue_center = []
            warp_blue_center = []

        if cone_centers[1].shape[0]:
            list_yell_center = cone_centers[1]
            warp_yell_center = self.perspective_warp_coordinates(list_yell_center,
                                                                 orig_im_shape,
                                                                 dst_size=(orig_im_shape[2], orig_im_shape[2]),
                                                                 src=src_trapezoide)
        else:
            list_yell_center = []
            warp_yell_center = []

        if cone_centers[2].shape[0]:
            list_oran_center = cone_centers[2]
            if cone_centers[3].shape[0]:
                list_oran_center = np.concatenate([list_oran_center, cone_centers[3]])
            warp_oran_center = self.perspective_warp_coordinates(list_oran_center,
                                                                 orig_im_shape,
                                                                 dst_size=(orig_im_shape[2], orig_im_shape[2]),
                                                                 src=src_trapezoide)
        elif cone_centers[3].shape[0]:
            list_oran_center = cone_centers[3]
            warp_oran_center = self.perspective_warp_coordinates(list_oran_center,
                                                                 orig_im_shape,
                                                                 dst_size=(orig_im_shape[2], orig_im_shape[2]),
                                                                 src=src_trapezoide)
        else:
            list_oran_center = []
            warp_oran_center = []
        # list_blue_center, list_yell_center, list_oran_center = self.split_cones(cone_centers, labels)

        if img_to_wrap is not None:
            img_wrap = self.perspective_warp_image(img_to_wrap,
                                                     orig_im_shape,
                                                     dst_size=(orig_im_shape[2], orig_im_shape[2]),
                                                     src=src_trapezoide,
                                                     wrap_img=True)
        else:
            img_wrap = np.zeros((orig_im_shape[2], orig_im_shape[2], 3))


        # # algoritmo para unir conos contiguos. Lo aplicamos sobre los conos en perspectiva
        # order_warp_blue_center = self.join_cones(warp_blue_center, unique_color='blue')
        # order_warp_yell_center = self.join_cones(warp_yell_center, unique_color='yell')
        # order_warp_oran_left_center, order_warp_oran_rigth_center = self.join_cones(warp_oran_center,
        #                                                                                 unique_color='oran')
        order_warp_blue_center = warp_blue_center
        order_warp_yell_center = warp_yell_center
        order_warp_oran_left_center, order_warp_oran_rigth_center = self.split_orange_cones(warp_oran_center)

        # calcular el centro
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

        # import matplotlib.pyplot as plt
        # fig = plt.figure(0)
        # plt.plot(list_blue_center[:, 0], list_blue_center[:, 1], 'o')
        # plt.plot(list_yell_center[:, 0], list_yell_center[:, 1], 'o')
        # plt.plot(list_oran_center[:, 0], list_oran_center[:, 1], 'o')
        # fig.canvas.draw()
        # # convert canvas to image
        # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # # img is rgb, convert to opencv's default bgr
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # # display image with opencv or any operation you like
        # cv2.imshow("plot", img)
        
        # fig = plt.figure(1)
        # plt.plot(warp_blue_center[:, 0], warp_blue_center[:, 1], 'o')
        # plt.plot(warp_yell_center[:, 0], warp_yell_center[:, 1], 'o')
        # plt.plot(warp_oran_center[:, 0], warp_oran_center[:, 1], 'o')
        # fig.canvas.draw()
        # # convert canvas to image
        # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # # img is rgb, convert to opencv's default bgr
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # # display image with opencv or any operation you like
        # cv2.imshow("plot_wrap", img)
        # cv2.waitKey(0)

        return [warp_blue_center, warp_yell_center, warp_oran_center], \
               [np.array(order_warp_blue_center), np.array(order_warp_yell_center),
                np.array(order_warp_oran_left_center), np.array(order_warp_oran_rigth_center)], \
               center, img_wrap

    '''
    def create_cone_map_legacy(self, cone_detections, cone_centers, aux_data=None):
        """
        Performs the cones detection task. The detection must include the bounding boxes and classification of each
        cone.
        :param img: (3D numpy array) Image to process.
        :param min_score_thresh: (float in [0., 1.]) Min score of confident on a detection.
        :param show_detections: (bool) If True: The image with detections id displayed. If False: no image is displayed.
        :param im_name: (string) Name of the detection image when show_detections=True
        :return: [ndarray, list] ndarray with detected bounding boxes and classification of each cone, list of auxiliar
                                data.
        """
        eagle_img, orig_im_shape = aux_data

        src_trapezoide = self.valTrackbars()  # Trapezoide a coger de la imagen original

        list_blue_center, list_yell_center, list_oran_center = self.split_cones_legacy(cone_centers)

        # Transformo las coordenadas a vista de águila
        warp_blue_center = self.perspective_warp_coordinates_legacy(list_blue_center,
                                                              eagle_img,
                                                              dst_size=(orig_im_shape[1], orig_im_shape[1]),
                                                              src=src_trapezoide)
        warp_yell_center = self.perspective_warp_coordinates_legacy(list_yell_center,
                                                              eagle_img,
                                                              dst_size=(orig_im_shape[1], orig_im_shape[1]),
                                                              src=src_trapezoide)

        warp_oran_center = self.perspective_warp_coordinates_legacy(list_oran_center,
                                                              eagle_img,
                                                              dst_size=(
                                                                  orig_im_shape[1], orig_im_shape[1]),
                                                              src=src_trapezoide)

        # algoritmo para unir conos contiguos. Lo aplicamos sobre los conos en perspectiva
        order_warp_blue_center = self.join_cones(warp_blue_center, unique_color='blue')
        order_warp_yell_center = self.join_cones(warp_yell_center, unique_color='yell')
        order_warp_oran_left_center, order_warp_oran_rigth_center = self.join_cones(warp_oran_center,
                                                                                        unique_color='oran')
        if len(warp_blue_center) > 1 and len(warp_yell_center) > 1:
            x1 = np.median(np.array(warp_blue_center)[:, 0])
            x2 = np.median(np.array(warp_yell_center)[:, 0])
            center = int((x2 - x1) / 2 + x1)
        elif len(order_warp_oran_left_center) > 1 and len(order_warp_oran_rigth_center) > 1:
            x1 = np.median(np.array(order_warp_oran_left_center)[:, 0])
            x2 = np.median(np.array(order_warp_oran_rigth_center)[:, 0])
            center = int((x2 - x1) / 2 + x1)
        else:
            center = 0.

        return [warp_blue_center, warp_yell_center, warp_oran_center], \
               [order_warp_blue_center, order_warp_yell_center, order_warp_oran_left_center, order_warp_oran_rigth_center], \
               center
    '''

    def perspective_warp_coordinates(self,
                                    coord_list,
                                    input_size,
                                    dst_size=(180, 180),
                                    src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                                    dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
        coord_list = np.array(coord_list)
        if coord_list.shape[0] > 0:
            # img_size = np.float32([(input_size[1], input_size[0])])
            img_size = np.float32([input_size[2], input_size[1]])
            src = src * img_size
            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result
            # again, not exact, but close enough for our purposes
            dst = dst * np.float32(dst_size)
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)

            c = np.float32(coord_list[np.newaxis, :])
            warped = np.int32(cv2.perspectiveTransform(c, M))
            return warped[0]
        return []

    def perspective_warp_image(self, image,
                                    input_size,
                                    dst_size=(180, 180),
                                    # src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]), # 
                                    src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                                    dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                                    wrap_img=True):

        img_size = np.float32([input_size[2], input_size[1]])
        src = src * img_size
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = dst * np.float32(dst_size)
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        if wrap_img:
            img_warped = cv2.warpPerspective(image, M, dst_size)
        else:
            img_warped = np.zeros((dst, dst, 3))
        return img_warped

    def perspective_warp_coordinates_legacy(self, coord_list,
                                     img,
                                     dst_size=(1280, 720),
                                     src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                                     dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
        coord_list = np.array(coord_list)
        if coord_list.shape[0] > 0:
            img_size = np.float32([(img.shape[1], img.shape[0])])
            src = src * img_size
            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result
            # again, not exact, but close enough for our purposes
            dst = dst * np.float32(dst_size)
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # Warp the image using OpenCV warpPerspective()
            # warped = cv2.warpPerspective(img, M, dst_size)

            c = np.float32(coord_list[np.newaxis, :])
            warped = np.int32(cv2.perspectiveTransform(c, M))
            return warped[0]
        return []

    def split_cones(self, cone_centers, cone_class):
        # TODO: reimplementar con numpy sin bucle
        b_center = []
        y_center = []
        o_center = []
        for center, clase in zip(cone_centers, cone_class):
            if clase == 0:
                b_center.append(center)
            elif clase == 1:
                y_center.append(center)
            elif clase == 2 or clase == 3:
                o_center.append(center)

        return b_center, y_center, o_center

    def split_cones_legacy(self, cone_centers):
        centers = np.array([c[0] for c in cone_centers])
        x = centers[:, 0]
        y = centers[:, 1]
        color = np.array([c[1] for c in cone_centers])
        blues = color == self.blue_color
        yellows = color == self.yellow_color
        oranges = color == self.orange_color

        b_center = centers[blues]
        y_center = centers[yellows]
        o_center = centers[oranges]

        return b_center, y_center, o_center

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
            list_oran_rigth = []
            for c in cone_centers:
                if c[0] > median:
                    list_oran_rigth.append(c)
                else:
                    list_oran_left.append(c)
            return np.array(list_oran_left), np.array(list_oran_rigth)
        return np.array([]), np.array([])

    def join_cones(self, cone_centers, unique_color=None):
        """
        :cone_centers list: [[(x_1, y_1), color_tag_1], ..., [(x_n, y_n), color_tag_n]]
        :unique_color string: 'blue'
                              'yell'
                              'oran'
                              Si se selecciona un único color para la lista de conos la entrada esperada para
                              cone_centers será: [[x_1, y_1], ... , [x_n, y_n]]
        """

        if unique_color is None:
            b_center, y_center, o_center = self.split_cones_legacy(cone_centers)
        else:
            b_center = []
            y_center = []
            o_center = []
            if unique_color == self.blue_color:
                b_center = cone_centers
            elif unique_color == self.yellow_color:
                y_center = cone_centers
            elif unique_color == self.orange_color:
                o_center = cone_centers

        list_blue_center = []
        list_yell_center = []
        list_oran_left = []
        list_oran_rigth = []

        if len(o_center) > 3:
            # Cogemos el cono que más abajo esté en la imagen
            left_orange, rigth_orange = self.take_two_first_oranges(o_center)

            l_index, _ = np.where(o_center == left_orange)
            r_index, _ = np.where(o_center == rigth_orange)

            l_index = l_index[0]
            r_index = r_index[0]
            list_oran_left.append(left_orange)
            list_oran_rigth.append(rigth_orange)

            if r_index > l_index:
                o_center = np.delete(o_center, r_index, axis=0)
                o_center = np.delete(o_center, l_index, axis=0)
            else:
                o_center = np.delete(o_center, l_index, axis=0)
                o_center = np.delete(o_center, r_index, axis=0)

            l_iter = 0
            r_iter = 0
            while len(o_center) > 1:
                # Calculamos distancia a otros conos y cogemos el más cercano


                try:
                    l_angle_condition = False
                    o_center_copy = np.copy(o_center)
                    while not l_angle_condition and len(o_center_copy) > 0:
                        l_distances = self.calc_distances(left_orange, o_center_copy)
                        l_index = np.argmin(l_distances)

                        if len(list_oran_left) > 0:
                            if len(list_oran_left) > 1:
                                v1 = list_oran_left[l_iter - 1] - list_oran_left[l_iter - 2]
                            else:
                                v1 = list_oran_left[l_iter - 1] - [list_oran_left[l_iter - 1][0],
                                                                   list_oran_left[l_iter - 1][1] + 10]
                            v2 = o_center_copy[l_index] - list_oran_left[l_iter - 1]
                            angle = self.angle_between(v1, v2)
                        else:
                            angle = 0.

                        radians_limit = np.pi / 4.  # np.deg2rad(45)
                        # radians_limit = 0.05
                        l_angle_condition = angle <= radians_limit

                        if l_angle_condition:
                            l_point_select = o_center_copy[l_index]
                        else:
                            o_center_copy = np.delete(o_center_copy, l_index, axis=0)
                except:
                    pass
                try:
                    r_angle_condition = False
                    o_center_copy = np.copy(o_center)
                    while not r_angle_condition and len(o_center_copy) > 0:
                        r_distances = self.calc_distances(rigth_orange, o_center_copy)
                        r_index = np.argmin(r_distances)

                        if len(list_oran_rigth) > 0:
                            if len(list_oran_rigth) > 1:
                                v1 = list_oran_rigth[r_iter - 1] - list_oran_rigth[r_iter - 2]
                            else:
                                v1 = list_oran_rigth[r_iter - 1] - [list_oran_rigth[r_iter - 1][0],
                                                                   list_oran_rigth[r_iter - 1][1] + 10]
                            v2 = o_center_copy[r_index] - list_oran_rigth[r_iter - 1]
                            angle = self.angle_between(v1, v2)
                        else:
                            angle = 0.

                        radians_limit = np.pi / 4.  # np.deg2rad(45)
                        # radians_limit = 0.05
                        r_angle_condition = angle <= radians_limit

                        if r_angle_condition:
                            r_point_select = o_center_copy[r_index]
                        else:
                            o_center_copy = np.delete(o_center_copy, r_index, axis=0)
                except:
                    pass

                if l_angle_condition:
                    # l_point_select = o_center_copy[l_index]
                    # left_orange = o_center[l_index]
                    left_orange = l_point_select
                    list_oran_left.append(left_orange)

                if r_angle_condition:
                    # r_point_select = o_center_copy[r_index]
                    # rigth_orange = o_center[r_index]
                    rigth_orange = r_point_select
                    list_oran_rigth.append(rigth_orange)


                if r_index != l_index:
                    if r_index > l_index:
                        o_center = np.delete(o_center, r_index, axis=0)
                        o_center = np.delete(o_center, l_index, axis=0)
                    else:
                        o_center = np.delete(o_center, l_index, axis=0)
                        o_center = np.delete(o_center, r_index, axis=0)

                l_iter += 1
                r_iter += 1
                if len(o_center) < 1 or l_iter > 40 or r_iter > 40:
                    break

        if len(b_center) > 0:
            self._join_check_cones(b_center, list_blue_center)

        if len(y_center) > 0:
            self._join_check_cones(y_center, list_yell_center)

        if unique_color == self.blue_color:
            return list_blue_center
        elif unique_color == self.yellow_color:
            return list_yell_center
        elif unique_color == self.orange_color:
            return list_oran_left, list_oran_rigth

        return list_blue_center, list_yell_center, list_oran_left, list_oran_rigth

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