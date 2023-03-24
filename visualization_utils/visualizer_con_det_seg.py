import cv2
import numpy as np

class Visualizer():
    def __init__(self):
        self.saved_frames = []

    def visualize(self, data, controls, fps, save_frames=False):
        """
        :param data: List of: [image, detections, cenital_map, y_hat, in_speed]
                     image: ndarray (h, w, channels)
                     detections: bboxes
                     cenital_map: eagle view image or coordinates map
                     y_hat: imagen segmentada (h, w, n_clases)
                     in_speed: float. Current car speed
        :param controls: List of [throttle, brake, steer, clutch, upgear, downgear, in_gear, in_rpm]
                     throttle: float [0, 1]
                     brake: float [0, 1]
                     steer: float [-1, 1]
                     clutch: float [0, 1]
                     upgear: bool
                     downgear: bool
                     in_gear: int. current gear
                     in_rpm: float. current rpm
        :param fps: int
        :param save_frames: bool. Allows store the resulting frame in a list to later create a video with save_in_video function
        """
        image, detections, cenital_map, y_hat, speed = data
        self.make_images(image, detections, cenital_map, y_hat, speed, controls, fps, save_frames=save_frames)


    def make_images(self, image, detections, cenital_map, y_hat, speed, controls, fps, dim=(1, 180, 320, 3), save_frames=False):
        centers, detections, labels = detections
        cenital_map, estimated_center = cenital_map
        throttle, brake, steer, clutch, upgear, downgear, gear, rpm = controls

        # Make images
        color_mask = np.argmax(y_hat[0], axis=2)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1080, 720))
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
                cv2.circle(res_image, (int(c[0]), int(c[1])), 1, (0, 0, 255), -1)


        cenital_img = np.zeros((dim[2], dim[2], 3)) * 255
        for c in cenital_map[0]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 3, (255, 0, 0), -1)
        for c in cenital_map[1]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 3, (0, 255, 255), -1)
        for c in cenital_map[2]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 3, (0, 0, 255), -1)
        for c in cenital_map[3]:
            cv2.circle(cenital_img, (int(c[0]), int(c[1])), 3, (100, 0, 255), -1)

        cv2.circle(cenital_img, (int(estimated_center), int(100)), 3, (0, 255, 0), -1)

        # cv2.imshow("eagle view", cenital_img)
        # cv2.imshow("color cones cones", res_image)
        # cv2.imshow("background", y_hat_resize_backg)
        # cv2.imshow("camera", image)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 255, 255)
        thickness = 1
        text = 'speed:   {:.2f}'.format(speed)
        image = cv2.putText(image, text, (10, 470), font, fontScale, color, thickness, cv2.LINE_AA)
        # text = 'throttle:   {:.4f}'.format(throttle)
        # image = cv2.putText(image, text, (10, 350), font, fontScale, color, thickness, cv2.LINE_AA)
        # te{:.4f}'.format(steer)
        # image = cv2.putText = 'brake:      {:.4f}'.format(brake)
        #         # image = cv2.putText(image, text, (10, 370), font, fontScale, color, thickness, cv2.LINE_AA)
        #         # text = 'steer:      xt(image, text, (10, 390), font, fontScale, color, thickness, cv2.LINE_AA)
        # text = 'clutch:     {:.4f}'.format(clutch)
        # image = cv2.putText(image, text, (10, 410), font, fontScale, color, thickness, cv2.LINE_AA)
        text = 'gear:     {:f}'.format(gear)
        image = cv2.putText(image, text, (10, 430), font, fontScale, color, thickness, cv2.LINE_AA)
        text = 'RPM:   {:f}'.format(rpm)
        image = cv2.putText(image, text, (10, 450), font, fontScale, color, thickness, cv2.LINE_AA)
        text = 'FPS:     {:.2f}'.format(fps)
        image = cv2.putText(image, text, (10, 490), font, fontScale, color, thickness, cv2.LINE_AA)

        ctr_img = self._controls_img(steer, throttle, brake, clutch)
        image[2:202, 10:210] = cv2.resize(cenital_img, (200, 200))
        image[220:331, 10:210] = cv2.resize(res_image, (200, 111))
        image[340:390, 10:210] = ctr_img
        cv2.imshow("Detections", image)

        cv2.waitKey(1)

        if save_frames:
            self.saved_frames.append(image)
        return image

    def _controls_img(self, steer, throttle, brake, clutch):
        text_steer =    'steer:  {:+.3f}'.format(steer)
        text_throttle = 'throttle: {:.3f}'.format(throttle)
        text_brake =    'brake:   {:.3f}'.format(brake)
        text_clutch =   'clutch:  {:.3f}'.format(clutch)

        ste_img = np.ones((5, 41, 3), dtype=np.uint8) * 255
        thr_img = np.ones((5, 41, 3), dtype=np.uint8) * 255
        brk_img = np.ones((5, 41, 3), dtype=np.uint8) * 255
        clutch_img = np.ones((5, 41, 3), dtype=np.uint8) * 255

        ctr_img = np.zeros((36, 87, 3), dtype=np.uint8)

        steer = np.int((steer + 1) / 2 * 41)
        throttle = np.int(np.clip(throttle, 0.0, 1.0) * 41)
        brake = np.int(np.clip(brake, 0.0, 1.0) * 41)
        # brake = np.int(np.clip(brake, 0.0, 1.0) * 41)
        clutch = np.int(np.clip(clutch, 0.0, 1.0) * 41)


        ste_img[:, steer:steer + 1, 1:3] = np.zeros((5, 1, 2), dtype=np.uint8)
        thr_img[:, :throttle, 1] = thr_img[:, :throttle, 1] * 0
        brk_img[:, :brake, 2] = brk_img[:, :brake, 2] * 0
        clutch_img[:, :clutch, 0] = clutch_img[:, :clutch, 0] * 0

        ctr_img[3:8, 43:84, :] = ste_img
        ctr_img[12:17, 43:84, :] = thr_img
        ctr_img[20:25, 43:84, :] = brk_img
        ctr_img[28:33, 43:84, :] = clutch_img


        ctr_img = cv2.resize(ctr_img, (200, 50))

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.4
        color = (255, 255, 255)
        thickness = 1
        ctr_img = cv2.putText(ctr_img, text_steer, (1, 10), font, fontScale, color, thickness, cv2.LINE_AA)
        ctr_img = cv2.putText(ctr_img, text_throttle, (1, 22), font, fontScale, color, thickness, cv2.LINE_AA)
        ctr_img = cv2.putText(ctr_img, text_brake, (1, 34), font, fontScale, color, thickness, cv2.LINE_AA)
        ctr_img = cv2.putText(ctr_img, text_clutch, (1, 46), font, fontScale, color, thickness, cv2.LINE_AA)

        return ctr_img

    def save_in_video(self, path, name):
        for i in range(len(self.saved_frames)):
            cv2.imwrite(path + name.format(i), self.saved_frames[i])
        cv2.destroyAllWindows()
