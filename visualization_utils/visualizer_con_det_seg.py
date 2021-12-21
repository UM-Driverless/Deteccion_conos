import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.saved_frames = []

    def visualize(self, data, controls, save_frames=False):
        image, detections, cenital_map, y_hat = data
        self.make_images(image, detections, cenital_map, y_hat, controls, save_frames=save_frames)


    def make_images(self, image, detections, cenital_map, y_hat, controls, dim=(1, 180, 320, 3), save_frames=False):
        centers, detections, labels = detections
        cenital_map, estimated_center = cenital_map
        throttle, brake, steer, clutch, upgear, downgear = controls

        # Make images
        color_mask = np.argmax(y_hat[0], axis=2)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        y_hat_resize_backg = y_hat[0, :, :, 4]

        res_image = np.zeros((180, 320, 3), dtype=np.uint8)

        res_image[:, :, 0][color_mask == 0] = 255

        res_image[:, :, 1][color_mask == 1] = 255
        res_image[:, :, 2][color_mask == 1] = 255

        res_image[:, :, 1][color_mask == 2] = 130
        res_image[:, :, 2][color_mask == 2] = 255

        res_image[:, :, 1][color_mask == 3] = 30
        res_image[:, :, 2][color_mask == 3] = 220

        # for det, lab in zip(detections[0], labels[0]):
        #     if lab == 0:
        #         color = (255, 0, 0)
        #     elif lab == 1:
        #         color = (0, 255, 255)
        #     elif lab == 2:
        #         color = (102, 178, 255)
        #     else:
        #         color = (102, 178, 255)
        #         # color = (0, 76, 153)
        # up_vertex = detections[0, :, 0, :]
        # down_vertex = detections[0, :, 1, :]
        # cone_bases = ((up_vertex[:, 0] + (down_vertex[:, 0] - up_vertex[:, 0]) / 2.).astype(dtype=np.int), down_vertex[:, 1])
        # for cb1, cb2 in zip(cone_bases[0], cone_bases[1]):
        #     image = cv2.circle(image, (cb1, cb2), radius=1, color=(0, 0, 255), thickness=5)

        # # visualizar find contours
        # # find contours in the binary image
        # aux = y_hat_resize_backg*255
        # aux = aux.astype('uint8')
        # ret, foreground = cv2.threshold(aux, int(0.3*255), int(1.*255), cv2.THRESH_BINARY_INV)
        # foreground = cv2.resize(foreground, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # foreground = cv2.erode(foreground, kernel)
        # contours, hierarchy = cv2.findContours(foreground, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        #
        # for c in contours:
        #     # calculate moments for each contour
        #     M = cv2.moments(c)
        #     if M["m00"] != 0:
        #         # calculate x,y coordinate of center
        #         cX = int(M["m10"] / M["m00"])
        #         cY = int(M["m01"] / M["m00"])
        #         cv2.circle(image, (cX, cY), 3, (0, 0, 255), -1)
        # cv2.imshow("foreground eroded", foreground)

        # Pintar centros de masa de los conos
        im_size = image.shape[:2]
        _scale2original_size = [im_size[0] / dim[1], im_size[1] / dim[2]]
        for i in range(4):
            for c in centers[i]:
                # cv2.circle(image,
                #            (int(c[0]*_scale2original_size[1]), int(c[1]*_scale2original_size[0])),
                #            3, (0, 0, 255), -1)
                cv2.circle(res_image, (int(c[0]), int(c[1])), 0, (0, 0, 255), 3)


        cenital_img = np.zeros((dim[2], dim[2], 3)) * 255
        for c in cenital_map[0]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 0, (255, 0, 0), 3)
        for c in cenital_map[1]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 0, (0, 255, 255), 3)
        for c in cenital_map[2]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 0, (0, 0, 255), 3)
        for c in cenital_map[3]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 0, (100, 0, 255), 3)

        cv2.circle(cenital_img, (int(estimated_center), int(100)), 0, (0, 255, 0), 3)

        # cv2.imshow("eagle view", cenital_img)
        # cv2.imshow("color cones cones", res_image)
        # cv2.imshow("background", y_hat_resize_backg)
        # cv2.imshow("camera", image)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 255, 255)
        thickness = 1
        text = 'throttle:   {:.4f}'.format(throttle)
        image = cv2.putText(image, text, (10, 350), font, fontScale, color, thickness, cv2.LINE_AA)
        text = 'brake:      {:.4f}'.format(brake)
        image = cv2.putText(image, text, (10, 370), font, fontScale, color, thickness, cv2.LINE_AA)
        text = 'steer:      {:.4f}'.format(steer)
        image = cv2.putText(image, text, (10, 390), font, fontScale, color, thickness, cv2.LINE_AA)
        text = 'clutch:     {:.4f}'.format(clutch)
        image = cv2.putText(image, text, (10, 410), font, fontScale, color, thickness, cv2.LINE_AA)
        text = 'upgear:     {:.4f}'.format(upgear)
        image = cv2.putText(image, text, (10, 430), font, fontScale, color, thickness, cv2.LINE_AA)
        text = 'downgear:   {:.4f}'.format(downgear)
        image = cv2.putText(image, text, (10, 450), font, fontScale, color, thickness, cv2.LINE_AA)

        image[10:330, 10:330] = cenital_img
        image[10:190, 973:1293] = res_image
        cv2.imshow("Detections", image)

        cv2.waitKey(1)

        if save_frames:
            self.saved_frames.append(image)
        return image

    def save_in_video(self, path, name):
        for i in range(len(self.saved_frames)):
            cv2.imwrite(path + name.format(i), self.saved_frames[i])
        cv2.destroyAllWindows()