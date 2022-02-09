from connection_utils.communication_controllers.camera_interface import CameraInterface
import pyzed.sl as sl

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
        cam = sl.Camera()
        cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 0)

        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = 0.15

        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit(1)

        runtime = sl.RuntimeParameters()
        runtime.sensing_mode = sl.SENSING_MODE.FILL

        err = cam.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            exit(1)

        return cam

    def get_frame(self):
        """
        Returns the current frame from the camera.
        :return: image (3D array)
        """
        mat_img = sl.Mat()
        self.cam.retrieve_image(mat_img)
        img = mat_img.get_data()
        return img
