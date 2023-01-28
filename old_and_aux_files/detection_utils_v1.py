import tensorflow as tf
import cv2
import time
from cone_detection.aux_def import *
from cone_detection.detector_base import ConeDetectorInterface

BLUE_COLOR = (255, 0, 0)
YELLOW_COLOR = (0, 237, 255)
ORANGE_COLOR = (0, 137, 255)
DARK_ORANGE_COLOR = (0, 87, 255)

class ConeDetector(ConeDetectorInterface):

    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = './cone_detection/saved_models/ResNet50_640x640_synt_2'

        super().__init__(checkpoint_path)
        self.detection_model, self.category_index = self._make_detect_model(self.checkpoint_path)
        self.detect = self._make_detection_func(self.detection_model)

        # HSVValues: lower_blues, upper_blues, lower_yellow, upper_yellow, lower_orange, upper_orange
        self.HSVvalues = [[73, 67, 27], [129, 255, 255], [29, 27, 57], [46, 255, 255], [5, 46, 32], [14, 180, 204]]
        self.blue_color = 'blue'
        self.yellow_color = 'yell'
        self.orange_color = 'oran'

    def detect_cones(self, img, show_detections=False, min_score_thresh=0.5, real_time=True, im_name='cone detections'):
        """
        Performs the cones detection task. The detection must include the bounding boxes and classification of each
        cone.
        :param input: (3D numpy array) Image to process.
        :param min_score_thresh: (float in [0., 1.]) Min score of confident on a detection.
        :param show_detections: (bool) If True: The image with detections id displayed. If False: no image is displayed.
        :param im_name: (string) Name of the detection image when show_detections=True
        :return: [ndarray, list] ndarray with detected bounding boxes and classification of each cone, list of auxiliar
                                data.
        """
        # cv2.imwrite('cones_image.png', img)
        img_resize = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
        detections = self.detect(img_resize)
        if show_detections:
            self._plot_det(img,
                     detections['detection_boxes'][0].numpy(),
                     detections['detection_classes'][0].numpy().astype(np.uint32)
                     + 1,
                     detections['detection_scores'][0].numpy(),
                     self.category_index,
                     min_score_thresh=min_score_thresh,
                     real_time=real_time,
                     im_name=im_name)

        # Cojo rectángulos de las detecciones y filtro los conos por color
        rectangles = np.array(detections["detection_boxes"])[0]
        image, cone_centers, cone_bases = self._color_filter_cones(img, rectangles, paint_rectangles=False,
                                                                      bgr=False)
        # Creo la imagen de vista de águila
        orig_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        eagle_img = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8) + np.uint8(
            orig_image * 0.5)

        # # Pinto los centros de los conos en las imágenes
        # image, = self._draw_centers(cone_centers, image)
        # eagle_img = self._draw_centers(cone_centers, eagle_img)

        return detections, [cone_centers, eagle_img]

    def _plot_det(self, image_np,
                 boxes,
                 classes,
                 scores,
                 category_index,
                 min_score_thresh=0.8,
                 real_time=True,
                 im_name='Detections'):
        # image_np_with_annotations = image_np.copy()
        # viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_annotations,
        #                                                     boxes,
        #                                                     classes,
        #                                                     scores,
        #                                                     category_index,
        #                                                     use_normalized_coordinates=True,
        #                                                     min_score_thresh=min_score_thresh)
        #
        # cv2.imshow(im_name, cv2.cvtColor(image_np_with_annotations, cv2.COLOR_RGB2BGR))
        viz_utils.visualize_boxes_and_labels_on_image_array(image_np,
                                                            boxes,
                                                            classes,
                                                            scores,
                                                            category_index,
                                                            use_normalized_coordinates=True,
                                                            min_score_thresh=min_score_thresh)

        cv2.imshow(im_name, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        if real_time:
            cv2.waitKey(1)
        else:
            cv2.waitKey(0)

    def _draw_centers(self, cone_centers, image):
        for center in cone_centers:
            c = center[0]
            color = center[1]
            if color == self.blue_color:
                color = BLUE_COLOR
            elif color == self.yellow_color:
                color = YELLOW_COLOR
            elif color == self.orange_color:
                color = ORANGE_COLOR

            image = cv2.circle(image, c, radius=2, color=color, thickness=2)
        return image

    def _make_detection_func(self, detection_model):
        # Again, uncomment this decorator if you want to run inference eagerly
        @tf.function(experimental_relax_shapes=True)
        def detect(input_tensor):
            """Run detection on an input image.

            Args:
              input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
                Note that height and width can_scripts be anything since the image will be
                immediately resized according to the needs of the model within this
                function.

            Returns:
              A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
                and `detection_scores`).
            """
            preprocessed_image, shapes = detection_model.preprocess(input_tensor)
            prediction_dict = detection_model.predict(preprocessed_image, shapes)
            return detection_model.postprocess(prediction_dict, shapes)

        def run_detect(img):
            input_tensor = tf.convert_to_tensor(np.array([img]), dtype=tf.float32)
            detections = detect(input_tensor)
            return detections
        return run_detect

    def _color_filter_cones(self, image, rectangles, paint_rectangles=True, bgr=True):
        image = np.array(image)
        im_h = image.shape[0]
        im_w = image.shape[1]
        if not bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        l_b = self.HSVvalues[0]
        u_b = self.HSVvalues[1]
        l_y = self.HSVvalues[2]
        u_y = self.HSVvalues[3]
        l_o = self.HSVvalues[4]
        u_o = self.HSVvalues[5]

        cone_center = []
        cone_base = []
        for i in range(15):
            # Sumo y resto 2 pixeles solo para mejorar la visualización
            ymin = int(rectangles[i][0] * im_h) - 2
            xmin = int(rectangles[i][1] * im_w) - 2
            ymax = int(rectangles[i][2] * im_h) + 2
            xmax = int(rectangles[i][3] * im_w) + 2
            if ymin < 0.0: ymin = 0
            if xmin < 0.0: xmin = 0
            if ymax > im_h: ymax = im_h
            if xmax > im_w: xmax = im_w
            rect_im = image[ymin:ymax, xmin:xmax, :]

            masked_blue = self.hsv_filter(rect_im, l_b, u_b)
            masked_yell = self.hsv_filter(rect_im, l_y, u_y)
            masked_oran = self.hsv_filter(rect_im, l_o, u_o)

            n_pixels = rect_im.shape[0] * rect_im.shape[1] * rect_im.shape[2]
            blue_value = np.sum(masked_blue) / n_pixels
            yell_value = np.sum(masked_yell) / n_pixels
            oran_value = np.sum(masked_oran) / n_pixels
            # cv2.imshow('masked blue', masked_blue)
            # cv2.imshow('masked yellow', masked_yell)
            # cv2.imshow('masked orange', masked_oran)
            # cv2.imshow("cone", rect_im)
            # cv2.waitKey(0)

            if blue_value > yell_value and blue_value > oran_value:
                color = BLUE_COLOR
                color_class = self.blue_color
            elif yell_value > blue_value and yell_value > oran_value:
                color = YELLOW_COLOR
                color_class = self.yellow_color
            elif oran_value > blue_value and oran_value > yell_value:
                color = ORANGE_COLOR
                color_class = self.orange_color

            if paint_rectangles:
                cv2.rectangle(image, (xmax, ymax), (xmin, ymin), color, 2)

            y = int(ymin + ((ymax-ymin)/2))
            x = int(xmin + ((xmax-xmin)/2))
            cone_center.append([(x, y), color_class])
            cone_base.append([(x, ymax), color_class])

        return image, cone_center, cone_base

    def _unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def _angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
                _angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                _angle_between((1, 0, 0), (1, 0, 0))
                0.0
                _angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _split_cones(self, cone_centers):
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
            b_center, y_center, o_center = self._split_cones(cone_centers)
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
                            angle = self._angle_between(v1, v2)
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
                            angle = self._angle_between(v1, v2)
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
                        angle = self._angle_between(v1, v2)
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
                pass            # last_point = b_center[index_b]
            if angle_condition:
                last_point = point_selected
                list_blue_center.append(last_point)
                b_center = np.delete(b_center, index_b, axis=0)
            iter += 1
            if len(b_center) < 1 or iter > 40:
                break

    def calc_distances(self, point, point_list):
        distances = [np.linalg.norm(point - p) for p in point_list]
        return distances

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

    def hsv_filter(self, img, l_values, u_values, bgr=True):
        if bgr:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower = np.array(l_values)
        upper = np.array(u_values)
        masked = cv2.inRange(hsv, lower, upper)
        return masked

    def _make_detect_model(self, checkpoint_path):
        cone_class_id = 1
        num_classes = 1
        category_index = {cone_class_id: {'id': cone_class_id, 'name': 'cone'}}

        # tf.keras.backend.clear_session()

        pipeline_config = '/home/shernandez/PycharmProjects/cone_detection/models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
        # Load pipeline config and build a detection model.
        #
        # Since we are working off of a COCO architecture which predicts 90
        # class slots by default, we override the `num_classes` field here to be just
        # one.
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        model_config.ssd.num_classes = num_classes
        model_config.ssd.freeze_batchnorm = True
        detection_model = model_builder.build(
            model_config=model_config, is_training=False)

        check = tf.train.Checkpoint(step=tf.Variable(1), model=detection_model)
        manager = tf.train.CheckpointManager(check, checkpoint_path, max_to_keep=None)
        check.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restaurado de {}".format(manager.latest_checkpoint))
        else:
            print("No se ha podido cargar el checkpoint")
        return detection_model, category_index


def main():
    data_path = "/home/shernandez/PycharmProjects/cone_detection/skidpad_640x640/"
    ground_truth_path = "/home/shernandez/PycharmProjects/cone_detection/skidpad_640x640/skidpad.csv"
    checkpoint_path = '/home/shernandez/PycharmProjects/cone_detection/saved_chpt_models/ResNet50_640x640_synt_2'


    im = cv2.imread('/home/shernandez/PycharmProjects/cone_detection/skidpad_640x640/0.jpg')
    img_names, _, _, _ = load_sync_gt(ground_truth_path, (im.shape[:-1]))

    img_list = []
    for name in img_names:
        img = cv2.imread(data_path + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)

    detector = ConeDetector(checkpoint_path)

    time_list = []
    t0 = time.time()
    for im in img_list:
        t_aux = time.time()
        detector.detect_in_image(im, plot=True, min_score_thresh=0.3, real_time=True)
        time_list.append(time.time() - t_aux)
    t1 = time.time()

    print("Time elapsed: ", t1 - t0, " FPS: ", len(img_list)/(t1-t0), "FPS mean: ", 1/np.mean(np.array(time_list)))  # CPU seconds elapsed (floating point)

# main()





