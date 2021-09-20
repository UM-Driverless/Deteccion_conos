from abc import ABCMeta, abstractmethod

class ConeProcessingInterface(object):
    def __init__(self):
        pass

    @abstractmethod
    def create_cone_map(self, cone_detections, cone_centers, aux_data=None):
        """
        Performs the cones detection task. The detection must include the bounding boxes and classification of each
        cone.
        :param cone_detections: (ndarray) Detections over cones.
        :param aux_data: (list) List of auxiliar data.
        :param show_detections: (bool) If True: The image with detections id displayed. If False: no image is displayed.
        :param im_name: (string) Name of the detection image when show_detections=True
        :return: [ndarray, list] ndarray with detected bounding boxes and classification of each cone, list of auxiliar
                                data.
        """