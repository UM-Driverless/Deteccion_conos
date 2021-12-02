import cv2
import numpy as np

class Visualizer:
    def visualize(self, data):
        image, detections, y_hat = data
        self.make_images(image, detections, y_hat)

    def make_images(self, image, detections, y_hat):
        detections, labels = detections
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

        for det, lab in zip(detections[0], labels[0]):
            if lab == 0:
                color = (255, 0, 0)
            elif lab == 1:
                color = (0, 255, 255)
            elif lab == 2:
                color = (102, 178, 255)
            else:
                color = (102, 178, 255)
                # color = (0, 76, 153)
        # up_vertex = detections[0, :, 0, :]
        # down_vertex = detections[0, :, 1, :]
        # cone_bases = ((up_vertex[:, 0] + (down_vertex[:, 0] - up_vertex[:, 0]) / 2.).astype(dtype=np.int), down_vertex[:, 1])
        # for cb1, cb2 in zip(cone_bases[0], cone_bases[1]):
        #     image = cv2.circle(image, (cb1, cb2), radius=1, color=(0, 0, 255), thickness=5)




        cv2.imshow("color cones cones", res_image)
        cv2.imshow("background", y_hat_resize_backg)
        cv2.imshow("Detections", image)
        cv2.waitKey(1)

        return image
