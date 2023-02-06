import sys
import pygame
from pygame import locals
from collections import deque

LEFT_JOY_X = 0
LEFT_JOY_Y = 1
LEFT_TRIGGER = 2
RIGHT_JOY_X = 3
RIGHT_JOY_Y = 4
RIGHT_TRIGGER = 5
A_BUTTON = 0
X_BUTTON = 2
Y_BUTTON = 3
B_BUTTON = 1
LB_BUTTON = 4
RB_BUTTON = 5
SELECT_BUTTON = 6
START_BUTTON = 7

pygame.init()
pygame.display.set_caption('game base')
screen = pygame.display.set_mode((500, 500), 0, 32)
clock = pygame.time.Clock()

pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
for joystick in joysticks:
    print(joystick.get_name())

max_pos = [450, 450]
min_pos = [0, 0]
red_square = pygame.Rect(0, 0, 50, 50)
green_square = pygame.Rect(50, 0, 50, 50)
blue_square = pygame.Rect(200, 100, 50, 50)
yell_square = pygame.Rect(200, 300, 50, 50)
mag_square_a = pygame.Rect(400, 0, 50, 50)
mag_square_x = pygame.Rect(450, 0, 50, 50)
mag_square_y = pygame.Rect(400, 50, 50, 50)
mag_square_b = pygame.Rect(450, 50, 50, 50)
mag_square_lb = pygame.Rect(400, 100, 50, 50)
mag_square_rb = pygame.Rect(450, 100, 50, 50)
mag_square_sel = pygame.Rect(400, 150, 50, 50)
mag_square_star = pygame.Rect(450, 150, 50, 50)




colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
my_square_color_a = colors[4]
my_square_color_x = colors[4]
my_square_color_y = colors[4]
my_square_color_b = colors[4]
my_square_color_lb = colors[4]
my_square_color_rb = colors[4]
my_square_color_sel = colors[4]
my_square_color_star = colors[4]
motion = [0, 0]


while True:

    screen.fill((0, 0, 0))

    pygame.draw.rect(screen, colors[0], red_square)
    pygame.draw.rect(screen, colors[1], green_square)
    pygame.draw.rect(screen, colors[2], blue_square)
    pygame.draw.rect(screen, colors[3], yell_square)

    pygame.draw.rect(screen, my_square_color_a, mag_square_a)
    pygame.draw.rect(screen, my_square_color_x, mag_square_x)
    pygame.draw.rect(screen, my_square_color_y, mag_square_y)
    pygame.draw.rect(screen, my_square_color_b, mag_square_b)
    pygame.draw.rect(screen, my_square_color_lb, mag_square_lb)
    pygame.draw.rect(screen, my_square_color_rb, mag_square_rb)
    pygame.draw.rect(screen, my_square_color_sel, mag_square_sel)
    pygame.draw.rect(screen, my_square_color_star, mag_square_star)


    if abs(motion[0]) < 0.1:
        motion[0] = 0
    if abs(motion[1]) < 0.1:
        motion[1] = 0
    # my_square.x += motion[0] * 10
    # my_square.y += motion[1] * 10


    axis_to_check = 0

    for event in pygame.event.get():
        if event.type == locals.JOYBUTTONDOWN:
            if event.button == A_BUTTON:
                print('Press A')
                my_square_color_a = colors[5]
            if event.button == X_BUTTON:
                print('Press X')
                my_square_color_x = colors[5]
            if event.button == Y_BUTTON:
                print('Press Y')
                my_square_color_y = colors[5]
            if event.button == B_BUTTON:
                print('Press B')
                my_square_color_b = colors[5]
            if event.button == LB_BUTTON:
                print('Press LB')
                my_square_color_lb = colors[5]
            if event.button == RB_BUTTON:
                print('Press RB')
                my_square_color_rb = colors[5]
            if event.button == SELECT_BUTTON:
                print('Press SELECT')
                my_square_color_sel = colors[5]
            if event.button == START_BUTTON:
                print('Press STAR')
                my_square_color_star = colors[5]
        if event.type == locals.JOYBUTTONUP:
            if event.button == A_BUTTON:
                my_square_color_a = colors[4]
            if event.button == X_BUTTON:
                my_square_color_x = colors[4]
            if event.button == Y_BUTTON:
                my_square_color_y = colors[4]
            if event.button == B_BUTTON:
                my_square_color_b = colors[4]
            if event.button == LB_BUTTON:
                my_square_color_lb = colors[4]
            if event.button == RB_BUTTON:
                my_square_color_rb = colors[4]
            if event.button == SELECT_BUTTON:
                my_square_color_sel = colors[4]
            if event.button == START_BUTTON:
                my_square_color_star = colors[4]

        if event.type == locals.JOYAXISMOTION:
            if event.axis == LEFT_JOY_X:
                print('Left JOY X', event.value)
                blue_square.x = ((event.value + 1) * 100) + 100
            if event.axis == LEFT_JOY_Y:
                blue_square.y = (event.value + 1) * 100
            if event.axis == RIGHT_JOY_X:
                yell_square.x = ((event.value + 1) * 100) + 100
            if event.axis == RIGHT_JOY_Y:
                yell_square.y = ((event.value + 1) * 100) + 250
            if event.axis == LEFT_TRIGGER:
                print('Left trigger', event.value)
                red_square.y = ((event.value + 1)/2.) * max_pos[0]
            if event.axis == RIGHT_TRIGGER:
                print('Rigth trigger', event.value)
                green_square.y = ((event.value + 1) / 2.) * max_pos[1]

            if event.axis < 2:
                motion[event.axis] = event.value
        if event.type == locals.JOYHATMOTION:
            print(event)
        if event.type == locals.JOYDEVICEADDED:
            joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
            for joystick in joysticks:
                print(joystick.get_name())
        if event.type == locals.JOYDEVICEREMOVED:
            joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
        if event.type == locals.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == locals.KEYDOWN:
            if event.key == locals.K_ESCAPE:
                pygame.quit()
                sys.exit()

    pygame.display.update()
    clock.tick(60)