from abc import ABCMeta, abstractmethod


class CameraInterface(object, metaclass=ABCMeta):
    @abstractmethod
    def init_camera(self):
        """
        Initialize the camera parameters.
        """

    @abstractmethod
    def get_frame(self):
        """
        Returns the current frame from the camera.
        :return: image (3D array)
        """
