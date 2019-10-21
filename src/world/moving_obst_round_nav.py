#!/usr/bin/env python

import rospy
import time
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ModelStates
from std_msgs.msg import String
import time


class Moving():
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.moving()

    # Only move obstacles if the robot is training in that specific module to save resources
    def moving(self):
        while not rospy.is_shutdown():
            # When training in the moving_obstacles module
            while True:
                obstacle = ModelState()
                model = rospy.wait_for_message('gazebo/model_states', ModelStates)
                for i in range(len(model.name)):
                    # Move obstacles
                    if model.name[i] == 'moving_obstacles_round' or \
                    model.name[i] == 'moving_obstacles_round_1' or \
                    model.name[i] == 'moving_obstacles_round_2':
                        obstacle.model_name = model.name[i]
                        obstacle.pose = model.pose[i]
                        obstacle.twist = Twist()
                        obstacle.twist.angular.z = 0.5
                        self.pub_model.publish(obstacle)


def main():
    rospy.init_node('moving_obstacle')
    moving = Moving()

if __name__ == '__main__':
    main()
