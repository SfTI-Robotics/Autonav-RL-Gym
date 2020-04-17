#!/usr/bin/env python

import rospy
import os
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float32
import torch
import torch.nn.functional as F
import gc
import torch.nn as nn
from collections import deque
from ppo.storage import Memory
from ppo.ppo_models import PPO
from env.training_environment import Env as train_env
from env.testing_environment import Env as test_env
from ppo_alg import PPO_agent
from ddpg_alg import DDPG_agent

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# dirPath = os.path.dirname(os.path.realpath(__file__))
env_module_id = 2
dirPath = '/tmp/env-{}/'.format(env_module_id)

MAX_STEPS = 500
MAX_EPISODES = 10001

if __name__ == '__main__':

    rospy.init_node('run_agent')

    Env = test_env
    load_ep = 0
    env = None
    agent = None

    # Choose correct environment
    if (sys.argv[1] == "train"):
        Env = train_env

    # arg 3 set load ep if specified
    if len(sys.argv) <= 3 + 2:
        load_ep = 0
    else:
        load_ep = int(sys.argv[3])

    # arg 2, agent PPO, DDPG, initialize Env(PPO) etc
    if (sys.argv[2] == "ppo"):
        env = Env("PPO", env_module_id)
        agent = PPO_agent(load_ep, env, MAX_STEPS, dirPath)
    elif (sys.argv[2] == "ddpg"):
        env = Env("DDPG", env_module_id)
        agent = DDPG_agent(load_ep, env, MAX_STEPS, dirPath)


    for ep in range(load_ep, MAX_EPISODES, 1):
        collision = 0
        goal = 0
        running_reward = 0
        ep_steps = 0
        state = env.reset()

        for step in range(MAX_STEPS):
            ep_steps += 1
            state, reward, collision, goal = agent.step(state,ep)
            running_reward += reward
            if (collision or goal or step == MAX_STEPS - 1):
                break

        env.logEpisode(running_reward, collision, goal, ep_steps)
        print("Episode " + str(ep))

        if (sys.argv[1] == "train"):
            agent.save(ep)

    print("Max episodes reached")
