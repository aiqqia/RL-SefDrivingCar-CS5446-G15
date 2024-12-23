import math
import os
import pygame
from PIL import Image
# from advanced_view.gauge import GaugeDraw
from pygame import gfxdraw
from random import randint
from pygame import Vector2

from vehicle import MAX_SPEED, VISION_F, VISION_W, VISION_B
import config
from config import ROAD_VIEW_OFFSET, INPUT_VIEW_OFFSET_X, INPUT_VIEW_OFFSET_Y


pygame.font.init()
font_28 = pygame.font.Font(os.path.join('./fonts/digitize.ttf'), 28)
font_60 = pygame.font.Font(os.path.join('./fonts/digitize.ttf'), 60)


class Point:
    # constructed using a normal tupple
    def __init__(self, point_t = (0,0)):
        self.x = float(point_t[0])
        self.y = float(point_t[1])

    # define all useful operators
    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))

    def __sub__(self, other):
        return Point((self.x - other.x, self.y - other.y))

    def __mul__(self, scalar):
        return Point((self.x*scalar, self.y*scalar))

    def __div__(self, scalar):
        return Point((self.x/scalar, self.y/scalar))

    def __len__(self):
        return int(math.sqrt(self.x**2 + self.y**2))

    # get back values in original tuple format
    def get(self):
        return self.x, self.y


# Defining Necessary colours
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
grey = pygame.Color(128, 128, 128)
yellow = pygame.Color(255, 255, 0, 128)
orange = pygame.Color(255, 140, 0, 128)
green = pygame.Color(0, 255, 0,100)

IMAGE_PATH = './images'

def draw_dashed_line(surface, color, start_pos, end_pos, width=1, dash_length=10, exclude_corners=True):
    start_pos = Vector2(start_pos)
    end_pos = Vector2(end_pos)
    displacement = end_pos - start_pos
    length = displacement.length()
    slope = displacement / length if length > 0 else Vector2(0, 0)

    for index in range(0, int(length / dash_length), 2):
        start = start_pos + (slope * index * dash_length)
        end = start + (slope * dash_length)
        pygame.draw.line(surface, color, start, end, width)

    if not exclude_corners:
        if (length % dash_length) < (dash_length / 2):
            pygame.draw.line(surface, color, start, end_pos, width)

def draw_dashed_line_delay(surface, color, start_pos, end_pos, width=1, dash_length=10, delay=0, exclude_corners=True):
    start_pos = Vector2(start_pos)
    end_pos = Vector2(end_pos)
    displacement = end_pos - start_pos
    length = displacement.length()
    slope = displacement / length if length > 0 else Vector2(0, 0)

    for index in range(0, int(length / dash_length), 2):
        start = start_pos + (slope * index * dash_length)
        end = start + (slope * dash_length)
        pygame.draw.line(surface, color, start, end, width)

    if not exclude_corners:
        if (length % dash_length) < (dash_length / 2):
            pygame.draw.line(surface, color, start, end_pos, width)

    if delay > 0:
        pygame.time.delay(delay)

def draw_basic_road(surface, speed):
    if not config.VISUALENABLED:    # works only if this is True
        return

    surface.fill(white)         # pygame surface object
    # Left most lane marking
    pygame.draw.line(surface, black, (ROAD_VIEW_OFFSET + 13, -10), (ROAD_VIEW_OFFSET + 13, 1000), 5)    # 1010 + 13, -10
    # Right most lane marking
    pygame.draw.line(surface, black, (ROAD_VIEW_OFFSET + 367, -10), (ROAD_VIEW_OFFSET + 367, 1000), 5)

    line_marking_offset = randint(0, 10)    # A random offset (line_marking_offset) is generated to add some variation to the position of the dashed lines
    for l in range(1, 7):
        draw_dashed_line(
            surface,
            grey,
            # Horizontal Positioning - ROAD_VIEW_OFFSET + l * 50 + 15, Vertical Offset Based on Speed - int((speed/(MAX_SPEED * 1.0)) * -1 * line_marking_offset)
            # speed / (MAX_SPEED * 1.0) calculates a ratio of the current speed to the maximum speed. 
            # If the car is going at maximum speed, this ratio will be 1 (or close to it). If it's stationary, the ratio will be 0.
            # This ratio is multiplied by -1 and line_marking_offset, which gives a negative vertical offset. The reason for the negative sign is likely to move 
            # the dashed lines up on the screen as the speed increases, creating the effect that they are moving down relative to the car.
            # At higher speeds, it looks like the dashed lines are passing by more quickly, while at lower speeds, they appear to move more slowly
            (ROAD_VIEW_OFFSET + l * 50 + 15, int((speed/(MAX_SPEED * 1.0)) * -1 * line_marking_offset)),
            (ROAD_VIEW_OFFSET + l * 50 + 15, 1000),
            width=1,
            dash_length=5
        )


def draw_road_overlay_safety(surface, lane_map):
    if not config.VISUALENABLED:
        return

    # Draw on surface
    for y in range(100):
        for x in range(7):
            if lane_map[y][x] != 0:
                pygame.draw.rect(surface, yellow, (ROAD_VIEW_OFFSET + x * 50 + 15 + 1, y * 10, 49, 10))
                pygame.draw.rect(surface, grey, (ROAD_VIEW_OFFSET + x * 50 + 15 + 1, y * 10, 49, 10), 1)


def draw_road_overlay_vision(surface, subject_car):
    if not config.VISUALENABLED:
        return

    # Draw on surface
    min_x = min(max(0, subject_car.lane - VISION_W - 1), 6)
    max_x = min(max(0, subject_car.lane + VISION_W - 1), 6)

    min_y = min(max(0, int(math.ceil(subject_car.y / 10.0)) - VISION_F - 1), 100)
    max_y = min(max(0, int(math.ceil(subject_car.y / 10.0)) + VISION_B - 1), 100)

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            pygame.draw.rect(surface, green, (ROAD_VIEW_OFFSET + x * 50 + 15 + 1, y * 10, 49, 10))
            pygame.draw.rect(surface, grey, (ROAD_VIEW_OFFSET + x * 50 + 15 + 1, y * 10, 49, 10), 1)


def control_car(target_car, keydown):
    a = ''
    if keydown == pygame.K_UP:
        target_car.move('A')
        a = 'A'
    elif keydown == pygame.K_DOWN:
        target_car.move('D')
        a = 'D'
    else:
        target_car.move('M')
        a = 'L'

    if keydown == pygame.K_LEFT:
        target_car.switch_lane('L')
        a = 'L'
    elif keydown == pygame.K_RIGHT:
        target_car.switch_lane('R')
        a = 'R'
    return a


def identify_free_lane(cars):
    # Initialize two lists of lanes (1-7), one for each zone
    lanes = [[n for n in range(1, 8)] for _ in range(2)]
    
    for car in cars:
        # Check cars in the upper zone (-170 to 0)
        if -170 <= car.y <= 0:
            if car.lane in lanes[0]:
                lanes[0].remove(car.lane)
            if car.switching_lane in lanes[0]:
                lanes[0].remove(car.switching_lane)
        # Check cars in the lower zone (930 to 1070)
        elif 930 <= car.y <= 1070:
            if car.lane in lanes[1]:
                lanes[1].remove(car.lane)
            if car.switching_lane in lanes[1]:
                lanes[1].remove(car.switching_lane)

    return lanes

def draw_inputs(surface, vision):
    vision_title = font_28.render("Vision:", False, (0, 0, 0))
    surface.blit(vision_title, (INPUT_VIEW_OFFSET_X - 10, INPUT_VIEW_OFFSET_Y))
    for y_i in range(len(vision)):
        for x_i, x in enumerate(vision[y_i]):
            pygame.draw.rect(surface, green if x != 0 else white,
                             (INPUT_VIEW_OFFSET_X + x_i * 10 + 80, INPUT_VIEW_OFFSET_Y + y_i * 10, 10, 10))
            pygame.draw.rect(surface, grey,
                             (INPUT_VIEW_OFFSET_X + x_i * 10 + 1 + 80, INPUT_VIEW_OFFSET_Y + y_i * 10, 10, 10), 1)


def draw_actions(surface, action):
    action_title = font_28.render("Action:", False, (0, 0, 0))
    action_text = font_28.render(action, False, (0, 0, 0))
    
    surface.blit(action_title, (INPUT_VIEW_OFFSET_X - 10, INPUT_VIEW_OFFSET_Y + 370))
    surface.blit(action_text, (INPUT_VIEW_OFFSET_X + 90, INPUT_VIEW_OFFSET_Y + 370))


class Score:
    def __init__(self, score=0):
        self.score = score

    def add(self):
        # self.score += 1
        self.score += 0.01

    def subtract(self):
        # self.score -= 1
        self.score -= 0.02

    def penalty(self):
        # Penalty over time
        self.score += config.CONSTANT_PENALTY

    def brake_penalty(self):
        self.score += config.EMERGENCY_BRAKE_PENALTY

    def action_mismatch_penalty(self):
        self.score += config.MISMATCH_ACTION_PENALTY

    def switching_lane_penalty(self):
        self.score += config.SWITCHING_LANE_PENALTY


def draw_gauge(surface, speed):
    # gauge = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
    speed_title = font_28.render("Speed:", False, (0, 0, 0))
    speed_value = font_60.render(str(int(speed)), False, (0, 0, 0))
    surface.blit(speed_title, (INPUT_VIEW_OFFSET_X - 10, 10))
    surface.blit(speed_value, (INPUT_VIEW_OFFSET_X - 10 + 80, 35))
    

def draw_score(surface, score):
    score_title = font_28.render("Score:", False, (0, 0, 0))
    score = font_60.render(str(int(score)), False, (0, 0, 0))
    surface.blit(score_title, (INPUT_VIEW_OFFSET_X - 10, 245))
    surface.blit(score, (INPUT_VIEW_OFFSET_X - 10 + 80, 240))
