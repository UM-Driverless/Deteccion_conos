import sys

import pygame

from pygame import locals
pygame.init()
pygame.display.set_caption('game base')
screen = pygame.display.set_mode((500, 500), 0, 32)
clock = pygame.time.Clock()

pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
for joystick in joysticks:
    print(joystick.get_name())

my_square = pygame.Rect(50, 50, 50, 50)
my_square_color = 0
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
motion = [0, 0]

while True:

    screen.fill((0, 0, 0))

    pygame.draw.rect(screen, colors[my_square_color], my_square)
    if abs(motion[0]) < 0.1:
        motion[0] = 0
    if abs(motion[1]) < 0.1:
        motion[1] = 0
    my_square.x += motion[0] * 10
    my_square.y += motion[1] * 10

    for event in pygame.event.get():
        if event.type == locals.JOYBUTTONDOWN:
            print(event)
            if event.button == 0:
                my_square_color = (my_square_color + 1) % len(colors)
        if event.type == locals.JOYBUTTONUP:
            print(event)
        if event.type == locals.JOYAXISMOTION:
            print(event)
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