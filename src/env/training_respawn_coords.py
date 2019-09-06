#!/usr/bin/env python

import random
import math

class module_empty():
    def __init__(self):
        self.name = "empty"
        self.model_x = -2
        self.model_y = 5
        self.goal_x = 0
        self.goal_y = 0
        self.bot_x = 0
        self.bot_y = 0

    def genBotPos(self):
        self.bot_x = random.uniform(-1.5, 1.5)
        self.bot_y = random.uniform(-1.5, 1.5)
        return self.bot_x + self.model_x, self.bot_y + self.model_y

    def genGoalPos(self):
        too_close = True

        while too_close:
            self.goal_x = random.uniform(-1.5, 1.5)
            self.goal_y = random.uniform(-1.5, 1.5)
            if (math.sqrt((self.goal_x - self.bot_x)**2 + (self.goal_y - self.bot_y)**2) >= 1):
                too_close = False

        return self.goal_x + self.model_x, self.goal_y + self.model_y

class module_move_away():
    def __init__(self):
        self.name = "move_away"
        self.model_x = 2
        self.model_y = 5
        self.goal_x = 0
        self.goal_y = 0
        self.bot_x = 0
        self.bot_y = 0

    def genBotPos(self):
        self.bot_x = random.uniform(-0.35, 0.35)
        self.bot_y = random.uniform(-0.5, -0.7)
        return self.bot_x + self.model_x, self.bot_y + self.model_y

    def genGoalPos(self):
        self.goal_x = random.uniform(-1.5, 1.5)
        self.goal_y = random.uniform(0.2, 1.5)
        return self.goal_x + self.model_x, self.goal_y + self.model_y


class module_left_right():
    def __init__(self):
        self.name = "left_right"
        self.model_x = -2
        self.model_y = 1
        self.goal_x = 0
        self.goal_y = 0
        self.bot_x = 0
        self.bot_y = 0

    def genBotPos(self):
        self.bot_x = 0
        self.bot_y = random.uniform(-1.5, 0)
        return self.bot_x + self.model_x, self.bot_y + self.model_y

    def genGoalPos(self):
        coefx = 1
        coefy = 1
        if random.random() < 0.5:
            coefx = -1
        if random.random() < 0.5:
            coefy = -1
        self.goal_x = random.uniform(1, 1.5) * coefx
        self.goal_y = 1.45 * coefy + 0.05 * coefy
        return self.goal_x + self.model_x, self.goal_y + self.model_y

class module_round_obstacle():
    def __init__(self):
        self.name = "round_obstacle"
        self.model_x = 2
        self.model_y = 1
        self.goal_x = 0
        self.goal_y = 0
        self.bot_x = 0
        self.bot_y = 0

    def genBotPos(self):
        self.bot_x = random.uniform(-1.4, 1.4)
        self.bot_y = -1.4
        return self.bot_x + self.model_x, self.bot_y + self.model_y

    def genGoalPos(self):
        self.goal_x = random.uniform(-1.4, 1.4)
        self.goal_y = 1.4
        return self.goal_x + self.model_x, self.goal_y + self.model_y

class module_static_obstacles():
    def __init__(self):
        self.name = "static_obstacles"
        self.model_x = -2
        self.model_y = -3
        self.goal_x = 0
        self.goal_y = 0
        self.bot_x = 0
        self.bot_y = 0

    def genBotPos(self):
        if random.random() > 0.5:
            self.bot_x = random.uniform(-1.5,1.5)
            self.bot_y = -1.55
        else:
            self.bot_x = random.uniform(-0.3,0.7)
            self.bot_y = -0.2
        
        return self.bot_x + self.model_x, self.bot_y + self.model_y

    def genGoalPos(self):
        self.goal_x = random.uniform(-1.5, 1.5)
        self.goal_y = 1.5
        return self.goal_x + self.model_x, self.goal_y + self.model_y

class module_moving_obstacles():
    def __init__(self):
        self.name = "moving_obstacles"
        self.model_x = 2
        self.model_y = -3
        self.goal_x = 0
        self.goal_y = 0
        self.bot_x = 0
        self.bot_y = 0

    def genBotPos(self):
        if random.random() > 0.5:
            self.bot_x = 0
            self.bot_y = 0
        else:
            self.bot_x = random.uniform(-1.5,1.5)
            self.bot_y = -1.5
        
        return self.bot_x + self.model_x, self.bot_y + self.model_y

    def genGoalPos(self):
        self.goal_x = random.uniform(-1.5, 1.5)
        self.goal_y = 1.5
        return self.goal_x + self.model_x, self.goal_y + self.model_y

class module_gate():
    def __init__(self):
        self.name = "gate"
        self.model_x = -2
        self.model_y = -7
        self.goal_x = 0
        self.goal_y = 0
        self.bot_x = 0
        self.bot_y = 0

    def genBotPos(self):
        self.bot_x = random.uniform(-1.5,1.5)
        self.bot_y = random.uniform(-1.4,-0.5)
        return self.bot_x + self.model_x, self.bot_y + self.model_y

    def genGoalPos(self):
        self.goal_x = random.uniform(-1.5,1.5)
        self.goal_y = random.uniform(0.5,1.4)
        return self.goal_x + self.model_x, self.goal_y + self.model_y