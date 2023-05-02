import numpy as np
import cv2
import matplotlib.pyplot as plt
from globals.globals import * # Global variables and constants, as if they were here

"""
def perspective_warp_coordinates(
                                coord_list,
                                input_size=IMAGE_RESOLUTION,
                                dst_size=IMAGE_RESOLUTION,
                                # src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                                # dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
                                src=np.float32([(0.4, 0.4), (0.6, 0.4), (1, 1), (0,1)]),
                                dst=np.float32([(0, 0), (1, 0), (1, 1), (0,1)])):
    '''    
    Takes coord_list
    '''
    coord_list = np.array(coord_list).transpose()
    if coord_list.shape[0] > 0:
        # img_size = np.float32([(input_size[1], input_size[0])])
        img_size = np.float32([input_size[1], input_size[0]])
        src = src * img_size
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = dst * np.float32(dst_size)
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

        c = np.float32(coord_list[np.newaxis, :])
        # warped = np.int32(cv2.perspectiveTransform(c, M))
        warped = cv2.perspectiveTransform(c, M)
        return warped[0]
    else:
        return []
"""
