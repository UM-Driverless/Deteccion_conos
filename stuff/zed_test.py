import pyzed.sl as sl
import cv2

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f'error: {err}')
    exit(-1)

runtime = sl.RuntimeParameters()
runtime.enable_depth = False # Deactivates de depth map calculation. We don't need it.

# Create an RGBA sl.Mat object
mat_img = sl.Mat()

while True:
    print('.')
    # Read ZED camera
    if (zed.grab(runtime) == sl.ERROR_CODE.SUCCESS): # Grab gets the new frame
        # recorded_times_0 = time.time()
        
        zed.retrieve_image(mat_img, sl.VIEW.LEFT) # Retrieve receives it and lets choose views and colormodes
        image = mat_img.get_data() # Creates np.array()4
        
        cv2.imshow('im',image)
        cv2.waitKey(10)
        # recorded_times_1 = time.time()
        # print(f'ZED read time: {recorded_times_1-recorded_times_0}')
