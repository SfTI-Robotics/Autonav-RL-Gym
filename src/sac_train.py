#!/usr/bin/env python
# Authors: Matthew Frost#

import rospy
import os
import numpy as np
import gc
import time
import sys
from util.replay_buffer import ReplayBuffer
from std_msgs.msg import Float32
from env.environment import Env
from sac.sac_models import Trainer

import time
# ---Directory Path---#
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
dirPath = os.path.dirname(os.path.realpath(__file__))

action_dim = 2
state_dim = 28
hidden_dim = 512
ACTION_V_MIN = 0  # m/s
ACTION_W_MIN = 0  # m/s
ACTION_V_MAX = 0.22  # m/s
ACTION_W_MAX = 0.22  # m/s

is_training = True
is_loading = False
load_ep = 0
max_episodes = 10001
max_steps = 400
rewards = []
batch_size = 128

replay_buffer_size = 100000
replay_buffer = ReplayBuffer(replay_buffer_size)

#guide = Trainer(state_dim, action_dim, ACTION_V_MIN, ACTION_V_MAX, ACTION_W_MIN, ACTION_W_MAX, replay_buffer, dirPath)
trainer = Trainer(state_dim, action_dim, ACTION_V_MIN, ACTION_V_MAX, ACTION_W_MIN, ACTION_W_MAX, replay_buffer, dirPath)

if is_loading:
    guide.load_models(load_ep)
else:
    load_ep = 0

if __name__ == '__main__':
    rospy.init_node('sac_train')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env()

    start_time = time.time()
    past_action = np.array([0., 0.])

    for ep in range(load_ep + 1, max_episodes, 1):
        done = False
        state = env.reset()
        print('Episode: ' + str(ep), 'Mem Buffer Size: ' + str(len(replay_buffer)))

        rewards_current_episode = 0

        for step in range(max_steps):

            state = np.float32(state)
	    #if len(replay_buffer) >= 2000:
		#print("trainer")
            	#action = trainer.get_action(state)
	    #else:
		#print "guide"
	    action = trainer.get_action(state)

            next_state, reward, done = env.step(action, past_action)
	
            past_action = action
		
            rewards_current_episode += reward
	    
            next_state = np.float32(next_state)
	    #intrinsic_reward = 0.1 * trainer.get_curiosity(state, action, next_state)
	    #print(reward)
	    #print(intrinsic_reward)
	    #reward += intrinsic_reward
	    #print("total reward: " + str(reward))
		
            replay_buffer.push(state, action, reward, next_state, done)
	    #if len(replay_buffer) == 25 and is_training and not is_loading:
	   #     print "training"
	#	for i in range (1000):
	#		trainer.soft_q_update(batch_size)

            if (len(replay_buffer) >= 2000 or (is_loading and len(replay_buffer) >= 2000)) and is_training:
                trainer.soft_q_update(batch_size)
		pass
            state = next_state
            if done:
                break
	
        print('reward per ep: ' + str(rewards_current_episode))
        rewards.append(rewards_current_episode)
        result = rewards_current_episode
        pub_result.publish(result)
        gc.collect()
        if ep % 20 == 0:
	    pass
            trainer.save_models(ep)
