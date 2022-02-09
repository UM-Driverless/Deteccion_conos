from controller_agent.agent_base import AgentInterface
import cv2
import numpy as np
import pygame

class Agent(AgentInterface):
    def __init__(self, im_width=1295, im_height=638, logger=None):
        super().__init__(logger=logger)

        # external control
        self.visualizer = visualize_pygame(im_width, im_height)
        self.mouse_init = (0., 0.)  # pygame.mouse.get_pos()
        self.width = im_width
        self.height = im_height

    def get_action(self, ref_point=None, img_center=None):
        """ Calcular los valores de control
        """
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_SPACE]:
            brake = 1.0
            return [0., brake, 0., 0., 0, 0]

        pos = pygame.mouse.get_pos()
        movement = (np.clip((pos[0] - self.mouse_init[0]) / self.width, -1., 1.) * 2.) - 1.
        print("movement: ", movement)
        pygame.event.pump()
        steer = movement

        throttle = 0.07
        brake = 0.0
        clutch = 0.0
        upgear = 0
        downgear = 0

        return [throttle, brake, steer, clutch, upgear, downgear]

class visualize_pygame:
    def __init__(self, im_width=1295, im_height=638):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("Mouse Controlling")
        self.width = im_width
        self.height = im_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.ticks = 30
        self.clock.tick(self.ticks)

        self.video = []

    def visualize(self, data):
        """
        :param data: List of base image, detections (bounding boxes of cones) and segmentations (imgae of segmentation)
        """
        image, detections, y_hat = data
        image = self.make_images(image, detections, y_hat)
        self.video.append(image)
        # Drawing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        self.screen.fill((0, 0, 0))
        self.screen.blit(im, (0, 0))
        pygame.display.flip()
        pygame.display.update()
        self.clock.tick(self.ticks)


    def save_images(self, path, name='frame_{:0>4d}.jpg'):
        for i in range(len(self.video)):
            cv2.imwrite(path + name.format(i), self.video[i])
        cv2.destroyAllWindows()

    def make_images(self, image, detections, y_hat):
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

        cv2.imshow("color cones cones", res_image)
        cv2.imshow("background", y_hat_resize_backg)

        for det in detections[0]:
            if det[0] == 0:
                color = (255, 0, 0)
            elif det[0] == 1:
                color = (0, 255, 255)
            elif det[0] == 2:
                color = (102, 178, 255)
            else:
                color = (102, 178, 255)
                # color = (0, 76, 153)
            a = tuple(det[1])
            b = tuple(det[2])
            image = cv2.rectangle(image, a, b, color=color, thickness=3)
        cv2.waitKey(1)

        return image
