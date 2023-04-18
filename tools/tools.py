import numpy as np
import cv2
import matplotlib.pyplot as plt
from globals.globals import * # Global variables and constants, as if they were here

def perspective_warp_coordinates(
                                coord_list,
                                input_size=IMAGE_RESOLUTION,
                                dst_size=IMAGE_RESOLUTION,
                                # src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                                # dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
                                src=np.float32([(0.4, 0.4), (0.6, 0.4), (1, 1), (0,1)]),
                                dst=np.float32([(0, 0), (1, 0), (1, 1), (0,1)])):
    '''
    TODO
        SIMPLIFY
        COMMENT
        MOVE TO UTILS (General utils that has everything)
    
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





# def perspective_warp_image(image,
#                            input_size,
#                            dst_size=(180, 180),
#                            # src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]), # 
#                            src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
#                            dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
#                            wrap_img=True):
#     '''
#     Takes an image, and makes a linear transformation to deform it
#     '''
#     img_size = np.float32([input_size[2], input_size[1]])
#     src = src * img_size
#     # For destination points, I'm arbitrarily choosing some points to be
#     # a nice fit for displaying our warped result
#     # again, not exact, but close enough for our purposes
#     dst = dst * np.float32(dst_size)
#     # Given src and dst points, calculate the perspective transform matrix
#     M = cv2.getPerspectiveTransform(src, dst)
#     # Warp the image using OpenCV warpPerspective()
#     if wrap_img:
#         img_warped = cv2.warpPerspective(image, M, dst_size)
#     else:
#         img_warped = np.zeros((dst, dst, 3))
#     return img_warped



# if img_to_wrap is not None:
#     img_wrap = self.perspective_warp_image(img_to_wrap,
#                                            orig_im_shape,
#                                            dst_size=(orig_im_shape[2], orig_im_shape[2]),
#                                            src=src_trapezoide,
#                                            wrap_img=True)
# else:
#     img_wrap = np.zeros((orig_im_shape[2], orig_im_shape[2], 3))