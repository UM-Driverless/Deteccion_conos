import tensorflow as tf
import cv2
import time
from cone_detection.aux_def import *

class ConeDetector:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.detection_model, self.category_index = self._make_detect_model(self.checkpoint_path)
        self.detect = self._make_detection_func(self.detection_model)

    def detect_in_image(self, img, plot=False, min_score_thresh=0.5, real_time=True, im_name='detections'):
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
        detections = self.detect(img)
        if plot:
            self.plot_det(img,
                     detections['detection_boxes'][0].numpy(),
                     detections['detection_classes'][0].numpy().astype(np.uint32)
                     + 1,
                     detections['detection_scores'][0].numpy(),
                     self.category_index,
                     min_score_thresh=min_score_thresh,
                     real_time=real_time,
                     im_name=im_name)
        return detections

    def plot_det(self, image_np,
                 boxes,
                 classes,
                 scores,
                 category_index,
                 min_score_thresh=0.8,
                 real_time=True,
                 im_name='Detections'):
        image_np_with_annotations = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_annotations,
                                                            boxes,
                                                            classes,
                                                            scores,
                                                            category_index,
                                                            use_normalized_coordinates=True,
                                                            min_score_thresh=min_score_thresh)

        cv2.imshow(im_name, cv2.cvtColor(image_np_with_annotations, cv2.COLOR_RGB2BGR))
        if real_time:
            cv2.waitKey(1)
        else:
            cv2.waitKey(0)

    def _make_detection_func(self, detection_model):
        # Again, uncomment this decorator if you want to run inference eagerly
        @tf.function(experimental_relax_shapes=True)
        def detect(input_tensor):
            """Run detection on an input image.

            Args:
              input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
                Note that height and width can be anything since the image will be
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





