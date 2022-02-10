import numpy as np

from connection_utils.communication_controllers.camera_interface import CameraInterface

class Camera(CameraInterface):
    """
    This clase controlls the ZED 2 camera.
    Here we can initialize the camera, get the current frame, depth and normal maps.
    """
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        self.cam = self.init_camera()

    def init_camera(self):
        return None

    def get_frame(self):
        """
        Returns the current frame from the camera.
        :return: image (3D array)
        """
        return np.zeros((300, 300, 3))
