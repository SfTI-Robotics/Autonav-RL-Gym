#!/usr/bin/env python

import rospy
import random
import time
import os
import math
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState 
from testing_respawn_coords import *
from tf.transformations import quaternion_from_euler


class Respawn():
    def __init__(self):

        # Tracking current module being trained in
        self.modules = [
            module_empty()#, 
            #module_left_right(),
            #module_move_away(),
            #module_round_obstacle(),
            #module_static_obstacles(),
            #module_moving_obstacles(),
            #module_gate()
            ]
        self.module_index = -1
        self.pub_module = rospy.Publisher('current_module', String, queue_size = 1)

        # Get goal box model
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('turtlebot3_ddpg/src/env',
                                                'turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()

        # Create initial goal pose
        self.goal_position = Pose()
        self.goal_position.position.x, self.goal_position.position.y = self.modules[0].genGoalPos()
        self.goal_position.position.z = 0.0
        self.goal_model_name = 'goal'

        # Create initial bot pose
        self.bot_position = Pose()
        self.bot_position.position.x, self.bot_position.position.y = self.modules[0].genBotPos()
        self.bot_position.position.z = 0.0
        self.bot_model_name = 'turtlebot3_burger'

        # Check if goal model exists, spawn if not (and set up pub/subs)
        self.check_model = False
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 1)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawnGoal()

    # Continually checks if goal already exists
    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    # If goal doesn't exist (checked continually by checkmodel), respawn it
    def respawnGoal(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.goal_model_name, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                break
            else:
                pass

    # Reposition goal and bot in new positions in next module
    def moduleRespawns(self, next_env=False):
	if next_env:
        	self.nextModule()
        self.repositionBot()
	#self.repositionTrack()
        return self.repositionGoal()

    # Fixes issue with racetrack position drifting on reset
    def repositionTrack(self):
        state_msg = ModelState()
        state_msg.model_name = "racetrack_static"
        state_msg.pose = Pose()
        state_msg.pose.position.x = 0.0
        state_msg.pose.position.y = 0.0
        state_msg.pose.position.z = 0.0
        rospy.wait_for_service('/gazebo/set_model_state')

        try:
            resp = self.pub_model.publish(state_msg)
            print(resp)
        except rospy.ServiceException, e:
            print "Reposition track: Service call failed: %s" % e

        rospy.wait_for_service('/gazebo/set_model_state')

        print("track repositioned")

    # Move the goal to its new position
    def repositionGoal(self):  

        self.goal_position.position.x, self.goal_position.position.y = self.modules[self.module_index].genGoalPos()

    	state_msg = ModelState()
    	state_msg.model_name = "goal"
    	state_msg.pose = self.goal_position
    	rospy.wait_for_service('/gazebo/set_model_state')

    	try:
        	resp = self.pub_model.publish(state_msg)
        	print(resp)
    	except rospy.ServiceException, e:
        	print "Reposition goal: Service call failed: %s" % e

    	rospy.wait_for_service('/gazebo/set_model_state')

        print("Goal repositioned")

        return self.goal_position.position.x, self.goal_position.position.y

    # Set the state to the next module to train in
    def nextModule(self):
        
        self.module_index = self.module_index + 1
        if (self.module_index >= len(self.modules)):
            self.pause_proxy()
            self.module_index = 0
        self.pub_module.publish(self.modules[self.module_index].name)
        print("Moving to " + self.modules[self.module_index].name + " module")

    # Move the goal to its new position & orientation
    def repositionBot(self):
        self.bot_position.position.x, self.bot_position.position.y = self.modules[self.module_index].genBotPos()

    	state_msg = ModelState()
    	state_msg.model_name = self.bot_model_name
    	state_msg.pose = self.bot_position
    	rospy.wait_for_service('/gazebo/set_model_state')

    	try:
        	resp = self.pub_model.publish(state_msg)
        	print(resp)
    	except rospy.ServiceException, e:
        	print "Reposition bot: Service call failed: %s" % e

    	rospy.wait_for_service('/gazebo/set_model_state')

        print("Bot repositioned")

        return self.bot_position.position.x, self.bot_position.position.y

    def currentModuleName(self):
        return self.modules[self.module_index].name
