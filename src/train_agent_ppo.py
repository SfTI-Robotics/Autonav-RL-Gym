#!/usr/bin/env python


import rospy
import os
import numpy as np
import gc
import time
import sys
from ppo.storage import Memory
from std_msgs.msg import Float32
from env.training_environment import Env
from ppo.ppo_models import PPO

import torch

# ---Directory Path---#
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
dirPath = os.path.dirname(os.path.realpath(__file__))

log_interval = 1           # print avg reward in the interval
max_episodes = 10000        # max training episodes
max_timesteps = 200        # max timesteps in one episode

update_timestep = 50      # update policy every n timesteps
action_std = 256            # constant std for action distribution (Multivariate Normal)
K_epochs = 20              # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr = 2e-4                # parameters for Adam optimizer
betas = (0.9, 0.999)

random_seed = None

load_ep = 0

#############################################

# creating environment
state_dim = 28
action_dim = 4
ACTION_V_MIN = 0  # m/s
ACTION_V_MAX = 0.4  # m/s

memory = Memory()
ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, ACTION_V_MIN, ACTION_V_MAX, eps_clip)


if __name__ == '__main__':
    rospy.init_node('ppo_train')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env()

    start_time = time.time()
    past_action = np.array([0., 0.])
    running_reward = 0
    avg_length = 0
    time_step = 0

    for ep in range(load_ep + 1, max_episodes, 1):
	done = 0
        collision_count = 0
        goal_count = 0
        ep_steps = 0
        state = env.reset()
        #print('Episode: ' + str(ep), 'Mem Buffer Size: ' + str(len(rollouts)))

        for step in range(max_timesteps):
	    time_step += 1
            ep_steps = ep_steps + 1
            action = ppo.select_action(state, memory)
	    #print("actual action = " + str(action))
            state, reward, done = env.step(action, past_action)

	    past_action = action
            # Saving reward:
            memory.rewards.append(reward)
	    memory.masks.append(float(done))
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
	    

            if done == 1:
                collision_count = 1
                break
            if done == 2:
                goal_count = 1
                break

        env.logEpisode(running_reward, collision_count, goal_count, ep_steps)
	
	# logging
        if ep % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(ep, avg_length, running_reward))
	    print('Timestep {}'.format(time_step))
            running_reward = 0
            avg_length = 0
	    

	    
	   

















