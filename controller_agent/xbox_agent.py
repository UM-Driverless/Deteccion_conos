from controller_agent.agent_base import AgentInterface
import numpy as np
import pygame
from pygame import locals

class AgentXboxController(AgentInterface):
    """
    Agent to test the actuators.

    Genrates a sin  wave to send to the actuators
    """
    def __init__(self, logger):
        super().__init__(logger=logger)

        self.gear = 0
        self.max_gear = 4
        self.min_gear = 0
        self.throttle = 0.
        self.brake = 1.
        self.steer = 0.
        self.clutch = 1.

        self.init_constants()
        self.init_pygame()

    def init_pygame(self):
        pygame.init()
        pygame.display.set_caption('game base')
        self.screen = pygame.display.set_mode((500, 500), 0, 32)
        self.clock = pygame.time.Clock()
        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
        for joystick in self.joysticks:
            print(joystick.get_name())

        self.max_pos = [450, 450]
        self.min_pos = [0, 0]
        self.red_square = pygame.Rect(0, 0, 50, 50)
        self.green_square = pygame.Rect(50, 0, 50, 50)
        self.blue_square = pygame.Rect(200, 100, 50, 50)
        self.yell_square = pygame.Rect(200, 300, 50, 50)
        self.mag_square_a = pygame.Rect(400, 0, 50, 50)
        self.mag_square_x = pygame.Rect(450, 0, 50, 50)
        self.mag_square_y = pygame.Rect(400, 50, 50, 50)
        self.mag_square_b = pygame.Rect(450, 50, 50, 50)
        self.mag_square_lb = pygame.Rect(400, 100, 50, 50)
        self.mag_square_rb = pygame.Rect(450, 100, 50, 50)
        self.mag_square_sel = pygame.Rect(400, 150, 50, 50)
        self.mag_square_star = pygame.Rect(450, 150, 50, 50)

        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        self.my_square_color_a = self.colors[4]
        self.my_square_color_x = self.colors[4]
        self.my_square_color_y = self.colors[4]
        self.my_square_color_b = self.colors[4]
        self.my_square_color_lb = self.colors[4]
        self.my_square_color_rb = self.colors[4]
        self.my_square_color_sel = self.colors[4]
        self.my_square_color_star = self.colors[4]

    def init_constants(self):
        self.LEFT_JOY_X = 0
        self.LEFT_JOY_Y = 1
        self.LEFT_TRIGGER = 2
        self.RIGHT_JOY_X = 3
        self.RIGHT_JOY_Y = 4
        self.RIGHT_TRIGGER = 5
        self.A_BUTTON = 0
        self.X_BUTTON = 2
        self.Y_BUTTON = 3
        self.B_BUTTON = 1
        self.LB_BUTTON = 4
        self.RB_BUTTON = 5
        self.SELECT_BUTTON = 6
        self.START_BUTTON = 7

    def get_action(self, program):
        """
        Calcula las acciones que hay que realizar

        :param program: ([int]) List with the selected actuators to try.
        :return: throttle, brake, steer, clutch, upgear, downgear
        """

        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, self.colors[0], self.red_square)
        pygame.draw.rect(self.screen, self.colors[1], self.green_square)
        pygame.draw.rect(self.screen, self.colors[2], self.blue_square)
        pygame.draw.rect(self.screen, self.colors[3], self.yell_square)

        pygame.draw.rect(self.screen, self.my_square_color_a, self.mag_square_a)
        pygame.draw.rect(self.screen, self.my_square_color_x, self.mag_square_x)
        pygame.draw.rect(self.screen, self.my_square_color_y, self.mag_square_y)
        pygame.draw.rect(self.screen, self.my_square_color_b, self.mag_square_b)
        pygame.draw.rect(self.screen, self.my_square_color_lb, self.mag_square_lb)
        pygame.draw.rect(self.screen, self.my_square_color_rb, self.mag_square_rb)
        pygame.draw.rect(self.screen, self.my_square_color_sel, self.mag_square_sel)
        pygame.draw.rect(self.screen, self.my_square_color_star, self.mag_square_star)

        upgear = False
        downgear = False

        for event in pygame.event.get():
            if event.type == locals.JOYBUTTONDOWN:
                if event.button == self.A_BUTTON:
                    self.my_square_color_a = self.colors[5]
                    upgear = self.gear < self.max_gear
                    self.gear = np.clip(self.gear + 1, self.min_gear, self.max_gear)
                if event.button == self.X_BUTTON:
                    self.my_square_color_x = self.colors[5]
                    downgear = self.gear > self.min_gear
                    self.gear = np.clip(self.gear - 1, self.min_gear, self.max_gear)
                if event.button == self.Y_BUTTON:
                    self.my_square_color_y = self.colors[5]
                if event.button == self.B_BUTTON:
                    self.my_square_color_b = self.colors[5]
                if event.button == self.LB_BUTTON:
                    self.my_square_color_lb = self.colors[5]
                if event.button == self.RB_BUTTON:
                    self.my_square_color_rb = self.colors[5]
                if event.button == self.SELECT_BUTTON:
                    self.my_square_color_sel = self.colors[5]
                if event.button == self.START_BUTTON:
                    self.my_square_color_star = self.colors[5]

            if event.type == locals.JOYBUTTONUP:
                if event.button == self.A_BUTTON:
                    self.my_square_color_a = self.colors[4]
                if event.button == self.X_BUTTON:
                    self.my_square_color_x = self.colors[4]
                if event.button == self.Y_BUTTON:
                    self.my_square_color_y = self.colors[4]
                if event.button == self.B_BUTTON:
                    self.my_square_color_b = self.colors[4]
                if event.button == self.LB_BUTTON:
                    self.my_square_color_lb = self.colors[4]
                if event.button == self.RB_BUTTON:
                    my_square_color_rb = self.colors[4]
                if event.button == self.SELECT_BUTTON:
                    self.my_square_color_sel = self.colors[4]
                if event.button == self.START_BUTTON:
                    self.my_square_color_star = self.colors[4]

            if event.type == locals.JOYAXISMOTION:
                if event.axis == self.LEFT_JOY_X:
                    self.blue_square.x = ((event.value + 1) * 100) + 100
                    self.steer = np.clip(event.value, -1 * program[0], 1. * program[0])
                if event.axis == self.LEFT_JOY_Y:
                    self.blue_square.y = (event.value + 1) * 100
                if event.axis == self.RIGHT_JOY_X:
                    self.yell_square.x = ((event.value + 1) * 100) + 100
                if event.axis == self.RIGHT_JOY_Y:
                    self.yell_square.y = ((event.value + 1) * 100) + 250
                    self.clutch = np.clip(((event.value - 0.1)), 0, 1 * program[0])


                if event.axis == self.LEFT_TRIGGER:
                    self.red_square.y = ((event.value + 1) / 2.) * self.max_pos[0]
                    self.brake = np.clip(((event.value + 1 - 0.1) / 2.), 0., 1 * program[0])
                if event.axis == self.RIGHT_TRIGGER:
                    self.green_square.y = ((event.value + 1) / 2.) * self.max_pos[1]
                    self.throttle = np.clip(((event.value + 1 - 0.1) / 2.), 0., 1 * program[0])

                pygame.display.update()
                self.clock.tick(100)
        return self.throttle, self.brake, self.steer, self.clutch, self.gear, upgear, downgear
