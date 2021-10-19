from abc import ABCMeta, abstractmethod

class ConeDetectorInterface(object, metaclass=ABCMeta):

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.detection_model = None

    @abstractmethod
    def detect_cones(self, input):
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
