import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import random
import io
# import imageio
import glob
import scipy.misc
import numpy as np
import pandas as pd
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
# from object_detection.utils import colab_utils
from object_detection.builders import model_builder



def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None,
                    opencv=False,
                    min_score_thresh=0.8,
                    plot=True):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=min_score_thresh)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  elif opencv:
    if plot:
        cv2.imshow('anotated', cv2.cvtColor(image_np_with_annotations, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
    return image_np_with_annotations
  else:
    if plot:
        plt.imshow(image_np_with_annotations)



def load_img(img_path, visualize=False):
    global train_image_dir, i
    matplotlib.use('TkAgg')
    # Load images and visualize
    train_image_dir = img_path
    train_images_np = []
    # for i in range(1, 61):
    #     image_path = os.path.join(train_image_dir, str(i) + '.jpg')
    #     train_images_np.append(load_image_into_numpy_array(image_path))

    images_paths = []
    img_numbers = []
    for image_path in glob.glob(train_image_dir + '/*.jpg'):
        print(image_path)
        images_paths.append(image_path)
        img_numbers.append(int(os.path.basename(image_path[:-4])))

    img_numbers = np.argsort(img_numbers)
    # for i in img_numbers:
    #     # print(images_paths[i])
    #     train_images_np.append(load_image_into_numpy_array(images_paths[i]))
    for path in images_paths:
        # print(images_paths[i])
        train_images_np.append(load_image_into_numpy_array(path))

    if visualize:
        plt.rcParams['axes.grid'] = False
        plt.rcParams['xtick.labelsize'] = False
        plt.rcParams['ytick.labelsize'] = False
        plt.rcParams['xtick.top'] = False
        plt.rcParams['xtick.bottom'] = False
        plt.rcParams['ytick.left'] = False
        plt.rcParams['ytick.right'] = False
        plt.rcParams['figure.figsize'] = [14, 7]
        # for idx, train_image_np in enumerate(train_images_np[30:36]):
        #     plt.subplot(2, 3, idx + 1)
        #     plt.imshow(train_image_np)
        # plt.show()
        for idx, train_image_np in enumerate(train_images_np[0:6]):
            plt.subplot(2, 3, idx + 1)
            plt.imshow(train_image_np)
        plt.show()
        # for idx, train_image_np in enumerate(train_images_np[0:6]):
        #     cv2.imshow('image', cv2.cvtColor(train_image_np, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(0)

    return train_images_np

def prepping_data(train_images_np, gt_boxes):
    global num_classes
    # By convention, our non-background classes start counting at 1.  Given
    # that we will be predicting just one class, we will therefore assign it a
    # `class id` of 1.
    duck_class_id = 1
    num_classes = 1
    category_index = {duck_class_id: {'id': duck_class_id, 'name': 'cone'}}
    # Convert class labels to one-hot; convert everything to tensors.
    # The `label_id_offset` here shifts all classes by a certain number of indices;
    # we do this here so that the model receives one-hot labels where non-background
    # classes start counting at the zeroth index.  This is ordinarily just handled
    # automatically in our training binaries, but we need to reproduce it here.
    label_id_offset = 1
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []
    for (train_image_np, gt_box_np) in zip(train_images_np, gt_boxes):
        # print('gt_box_np: ', gt_box_np)
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
            train_image_np, dtype=tf.float32), axis=0))
        gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(
            np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)
        gt_classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, num_classes))
    # print('Done prepping data.')

    return category_index, \
           train_image_tensors, gt_classes_one_hot_tensors, \
           zero_indexed_groundtruth_classes, \
           gt_box_tensors



def check_img_gt(gt_boxes, train_images_np, category_index):
    global i
    # dummy_scores = np.array([1.0], dtype=np.float32)  # give boxes a score of 100%
    plt.figure(figsize=(30, 15))
    count = 0
    for idx in range(0, 6):
        dummy_scores = np.array([1.0 for i in range(gt_boxes[idx].shape[0])], dtype=np.float32)
        plt.subplot(2, 3, count + 1)
        plot_detections(
            train_images_np[idx],
            gt_boxes[idx],
            np.ones(shape=[gt_boxes[idx].shape[0]], dtype=np.int32),
            dummy_scores, category_index)
        count += 1
    plt.show()


def run_test(detection_model, category_index, data_path=None, test_img_names=None, test_gt=None, batch_size=8):

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

    get_next_text_batch = load_batches_mit_img(data_path, test_img_names, test_gt, batch_size=batch_size)

    label_id_offset = 1

    # Note that the first frame will trigger tracing of the tf.function, which will
    # take some time, after which inference should be fast.
    for n in range(0, test_img_names.shape[0], batch_size):
        test_images, gt_train_boxes, _ = get_next_text_batch()

        input_tensor = tf.convert_to_tensor(test_images, dtype=tf.float32)
        detections = detect(input_tensor)
        for i in range(test_images.shape[0]):

            plot_detections(
                test_images[i],
                detections['detection_boxes'][i].numpy(),
                detections['detection_classes'][i].numpy().astype(np.uint32)
                + label_id_offset,
                detections['detection_scores'][i].numpy(),
                category_index, figsize=(15, 20),
                opencv=True,
                min_score_thresh=0.3)  # image_name="gif_frame_" + ('%02d' % i) + ".jpg"
            plt.show()

def test(detection_model, model_im_w, model_im_h, batch_size):

    # Again, uncomment this decorator if you want to run inference eagerly
    @tf.function(experimental_relax_shapes=True)
    def run(image_tensors,
            groundtruth_boxes_list,
            groundtruth_classes_list):
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
        shapes = tf.constant(batch_size * [[model_im_w, model_im_h, 3]], dtype=tf.int32)
        detection_model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list,
                                            groundtruth_classes_list=groundtruth_classes_list)
        preprocessed_images = tf.concat([detection_model.preprocess(image_tensor)[0]
                                         for image_tensor in image_tensors], axis=0)
        prediction_dict = detection_model.predict(preprocessed_images, shapes)
        losses_dict = detection_model.loss(prediction_dict, shapes)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
        return total_loss
    return run

def run_skidpad_test(detection_model, category_index, data_path=None, test_img_names=None, test_gt=None, batch_size=8):

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

    img_list = []
    label_id_offset = 1
    for name in test_img_names:
        img = cv2.imread(data_path + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(np.array([img]), dtype=tf.float32)
        detections = detect(input_tensor)

        img_list.append(plot_detections(img,
                                        detections['detection_boxes'][0].numpy(),
                                        detections['detection_classes'][0].numpy().astype(np.uint32)
                                        + label_id_offset,
                                        detections['detection_scores'][0].numpy(),
                                        category_index, figsize=(15, 20),
                                        opencv=True,
                                        min_score_thresh=0.3,
                                        plot=True))

    size = img.shape[:-1]
    out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for img in img_list:
        out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release()

def load_mit_gt(path):

    matplotlib.use('TkAgg')
    gt_boxes = pd.read_csv(path)

    gt_boxes = gt_boxes.to_numpy()[1:]

    cone_class_id = 1
    num_classes = 1
    category_index = {cone_class_id: {'id': cone_class_id, 'name': 'cone'}}

    list_name = []
    list_im_w = []
    list_im_h = []
    # img = cv2.imread('MIT_DATASET/vid_40_frame_497.jpg')
    # img_aux = np.copy(img)
    gt_boxes_aux = []
    boxes = []
    for idx in range(gt_boxes.shape[0]):
        boxes = []
        im_w = int(gt_boxes[idx][2])
        im_h = int(gt_boxes[idx][3])
        # img = cv2.imread('MIT_DATASET/' + gt_boxes[idx][0])

        for box in gt_boxes[idx][5:]:
            if isinstance(box, str):
                box = box[1:-1].split(',')
                x = int(box[0])
                y = int(box[1])
                h = int(box[2])
                w = int(box[3])
                boxes.append([y / im_h, x / im_w, (y + h) / im_h, (x + w) / im_w])

                # image = cv2.rectangle(img, (x, y), (x + w, y + h),
                #                       (0, 0, 255), 5)

        # Eliminar las que no tienen detecciones
        if boxes:
            list_name.append(gt_boxes[idx][0])
            list_im_w.append(int(gt_boxes[idx][2]))
            list_im_h.append(int(gt_boxes[idx][3]))

            boxes = np.array(boxes)
            gt_boxes_aux.append(boxes)

            # cv2.imshow('img', image)
            # cv2.waitKey(0)

            # dummy_scores = np.array([1.0 for i in range(boxes.shape[0])], dtype=np.float32)
            # plot_detections(
            #     cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            #     boxes,
            #     np.ones(shape=[boxes.shape[0]], dtype=np.int32),
            #     dummy_scores, category_index)
            # plt.show()

    list_name = np.array(list_name)
    list_im_w = np.array(list_im_w)
    list_im_h = np.array(list_im_h)
    gt_boxes_aux = np.array(gt_boxes_aux)

    return list_name, gt_boxes_aux, list_im_w, list_im_h

def load_sync_gt(path, im_size):

    matplotlib.use('TkAgg')
    gt_boxes = pd.read_csv(path)

    gt_boxes = gt_boxes.to_numpy()[:]

    im_h = im_size[0]
    im_w = im_size[1]
    cone_class_id = 1
    num_classes = 1
    category_index = {cone_class_id: {'id': cone_class_id, 'name': 'cone'}}

    list_name = []
    list_im_w = []
    list_im_h = []

    gt_boxes_aux = []
    aux_boxes = []
    aux_img_name = 0
    for idx in range(gt_boxes.shape[0]):
        boxes = []
        # gt_boxes_split = gt_boxes[idx][0].split(' ')
        gt_boxes_split = gt_boxes[idx]
        if int(gt_boxes_split[0]) != aux_img_name:
            list_name.append(str(aux_img_name)+'.jpg')
            gt_boxes_aux.append(np.array(aux_boxes))
            aux_img_name = int(gt_boxes_split[0])

            aux_boxes = []

        x0 = int(gt_boxes_split[2])
        y0 = int(gt_boxes_split[3])
        x1 = int(gt_boxes_split[4])
        y1 = int(gt_boxes_split[5])

        aux_boxes.append([y0 / im_h, x0 / im_w, y1 / im_h, x1 / im_w])
        # img = cv2.imread('MIT_DATASET/' + gt_boxes[idx][0])

    list_name = np.array(list_name)
    list_im_w = np.array(list_im_w)
    list_im_h = np.array(list_im_h)
    gt_boxes_aux = np.array(gt_boxes_aux)

    return list_name, gt_boxes_aux, list_im_w, list_im_h


def load_mit_img(folder_path, list_name, n_img_to_load=None):
    images = []
    i = 0
    for name in list_name:
        if i % 100 == 0:
            print('Loaded ', i, ' images of ', list_name.shape[0])

        img = cv2.imread(folder_path + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

        if n_img_to_load is not None:
            if i >= n_img_to_load:
                break
        i += 1
    return np.array(images)

def load_batches_mit_img(folder_path, list_img_names, gt_boxes, batch_size, shuffle=True, preprocess=False):
        class batch_iterator:
            def __init__(self, folder_path, list_img_names, gt_boxes , batch_size, dataset_size, shuffle):
                self.folder = folder_path
                self.img_names = list_img_names
                self.gt_boxes = gt_boxes
                self.iterador = 0
                self.batch_size = batch_size
                self.dataset_size = dataset_size
                self.shuffle = shuffle

            def get_it_bounds(self):
                init = self.iterador
                self.iterador = self.iterador + self.batch_size
                if self.iterador > self.dataset_size:
                    self.iterador = 0
                    fin = self.dataset_size

                    random_permutation = np.random.choice(self.img_names.shape[0], self.img_names.shape[0],
                                                          replace=False)
                    self.gt_boxes = self.gt_boxes[random_permutation]
                    self.img_names = self.img_names[random_permutation]
                else:
                    fin = self.iterador

                return init, fin

        iterator = batch_iterator(folder_path, list_img_names, gt_boxes, batch_size, list_img_names.shape[0], shuffle)

        def get_next_batch():
            init, fin = iterator.get_it_bounds()
            images = []
            for name in iterator.img_names[init:fin]:
                img = cv2.imread(iterator.folder + name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if preprocess:
                    img = preproc_im(img, '')
                images.append(img)

            return np.array(images), iterator.gt_boxes[init:fin], init
        return get_next_batch

def preproc_im(img, labels):
    color = np.random.rand()
    contrast = np.random.rand()
    bright = np.random.rand()
    blur = np.random.rand()
    noise = np.random.rand()

    # if blur > 0.:
    #     cv2.imshow('original', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if color > 0.75:
        img[:, :, 0] = np.clip(img[:, :, 0] + np.random.normal(0.0, 7.0), 0, 255).astype(np.uint8)
        img[:, :, 1] = np.clip(img[:, :, 1] + np.random.normal(0.0, 7.0), 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(img[:, :, 2] + np.random.normal(0.0, 7.0), 0, 255).astype(np.uint8)
        # cv2.imshow('coloured', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if contrast > 0.75:
        img = np.clip(img * (np.random.rand()*2 + 0.2), 0, 255).astype(np.uint8)
        # cv2.imshow('contrast', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if bright > 0.75:
        img = np.clip(img + np.random.normal(0.0, 40.0), 0, 255).astype(np.uint8)
        # cv2.imshow('brightness', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if blur > 0.75:
        if np.random.rand() > 0.5:
            k = np.random.randint(3) * 2 + 1
            img = cv2.GaussianBlur(img, (k, k), 0)
            # cv2.imshow('blurred', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            k = np.random.randint(3) * 2 + 1
            img = cv2.medianBlur(img, k)
            # cv2.imshow('median', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if noise > 0.90:
        img = np.clip(img + np.random.normal(0.0, 8.0, img.shape), 0, 255).astype(np.uint8)
        k = np.random.randint(3) * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
        # cv2.imshow('median', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


    # cv2.waitKey(0)

    return img




