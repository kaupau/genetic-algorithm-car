import pygame
from pygame.math import Vector2
from pygame.locals import *
import random
import nn
import time

GAME_TICKS = 60
SIZE = 10
FOCAL_DIST = 40
INITIAL_POS = [40, 40]
ROAD = 0
STEP_DIST = 5
ITERATION_TIME = 1

class Phenotype:
    def __init__(self, weights):
        self.weights = weights
        self.fitness = 0

        self.sprite = pygame.image.load("car.png")
        self.sprite = pygame.transform.scale(self.sprite, (30, 30))
        self.pos = INITIAL_POS.copy()
        # declare NN

    def display(self, window):
        # showing fitness score
        font = pygame.font.Font('freesansbold.ttf', 10)
        text = font.render(str(self.fitness), True, (0, 0, 128))
        textRect = text.get_rect()
        textRect.center = self.pos

        window.blit(text, textRect)
        window.blit(self.sprite, tuple(self.pos))

    def observe(self, window):
        # print(self.pos)
        try:
            current_pixel = window.get_at((self.pos[1], self.pos[1]))[0]
            if current_pixel != ROAD:
                self.fitness += 50


            # Left
            left_pixel = 1 if window.get_at((self.pos[0], self.pos[1] - FOCAL_DIST))[0] == ROAD else 0
            # Right
            right_pixel = 1 if window.get_at((self.pos[0], self.pos[1] + FOCAL_DIST))[0] == ROAD else 0
            # Up
            up_pixel = 1 if window.get_at((self.pos[0] - FOCAL_DIST, self.pos[1]))[0] == ROAD else 0
            # Down
            down_pixel = 1 if window.get_at((self.pos[0] + FOCAL_DIST, self.pos[1]))[0] == ROAD else 0

            nn_input = [left_pixel, right_pixel, up_pixel, down_pixel]
            # print(nn_input)
            return nn_input
        except:
            self.fitness += 100
            return [random.randrange(0, 2)]*4

    def evaluate(self, input):
        # print("propogating inputs", input)
        nn_output = nn.forward_propagate(self.weights, input)
        highest_probability = max(nn_output)
        nn_output = nn_output.index(highest_probability) + 1
        # print("got output", nn_output)
        return nn_output
        # return random.randrange(1, 5)

    def action(self, output):
        # Left
        if output==1:
            self.pos[0] -= STEP_DIST
            # print("LEFT")
        # Right
        if output==2:
            self.pos[0] += STEP_DIST
            self.fitness -= 30
            # print("RIGHT")
        # Up
        if output==3:
            self.pos[1] -= STEP_DIST
            # print("UP")
        # Down
        if output==4:
            self.pos[1] += STEP_DIST
            # print("DOWN")

class Game:
    def __init__(self, population):
        self.clock = pygame.time.Clock()
        self.ticks = GAME_TICKS
        self.population = population
        self.timer = 0

        pygame.init()
        pygame.display.set_caption("Self Driving Car Simulation")
        window = pygame.display.set_mode((1200, 400))

        track = pygame.image.load("track.png")

        # time.sleep(5)

        while True:
            window.blit(track, (0, 0))

            for phenotype in population:
                nn_input = phenotype.observe(window)
                nn_decision = phenotype.evaluate(nn_input)
                phenotype.action(nn_decision)
                phenotype.display(window)

            # Update the display
            pygame.display.update()
            # Update the clock (Called once per frame)
            self.clock.tick(self.ticks)
            self.timer += 1
            if(self.timer/60 > ITERATION_TIME):
                break

    def get_scores(self):
        scores = []
        for p in self.population:
            p.fitness -= p.pos[0]*1.5
            print("pos", p.pos)
            scores.append( p.fitness )
        print(scores)
        return scores