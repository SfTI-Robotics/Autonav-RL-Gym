#!/usr/bin/env python

import random
import math

class module_empty():
    def __init__(self):
        self.name = "empty"

    def genBotPos(self):
        return 0, 0

    def genGoalPos(self):
        return 0, 6.5

class module_move_away():
    def __init__(self):
        self.name = "move_away"

    def genBotPos(self):
        return -3, 15

    def genGoalPos(self):
        return 3, 17.8


class module_left_right():
    def __init__(self):
        self.name = "left_right"

    def genBotPos(self):
        return 0, 7

    def genGoalPos(self):
        return -5,11.5

class module_round_obstacle():
    def __init__(self):
        self.name = "round_obstacle"

    def genBotPos(self):
        return 3, 19.5

    def genGoalPos(self):
        return 3, 24.75

class module_static_obstacles():
    def __init__(self):
        self.name = "static_obstacles"

    def genBotPos(self):
        return 3, 26

    def genGoalPos(self):
        return 3, 30.3

class module_moving_obstacles():
    def __init__(self):
        self.name = "moving_obstacles"

    def genBotPos(self):
        return 3, 32

    def genGoalPos(self):
        return -2, 32.3

class module_gate():
    def __init__(self):
        self.name = "gate"
    def genBotPos(self):
        return -5, 33.5

    def genGoalPos(self):
        return -5, 36.5
