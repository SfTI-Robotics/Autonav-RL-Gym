#!/usr/bin/env python

import rospy
import time
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ModelStates
from std_msgs.msg import String
import time

GATE_MOVE_SPEED = 1 # m/s
GATE_WAIT_TIME = 8 # s


class Moving():
    def __init__(self):
        self.current_module = "empty"
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.sub_module = rospy.Subscriber('/current_module', String, self.updateModule)
        self.gate_dir = 1
        self.moving()

    # Keep track of which module currently training in
    def updateModule(self, new_module):
        self.current_module = new_module.data
        self.gate_dir = 1

    # Only move obstacles if the robot is training in that specific module to save resources
    def moving(self):
        while not rospy.is_shutdown():
            # When training in the moving_obstacles module
            while self.current_module == "moving_obstacles":
                obstacle = ModelState()
                model = rospy.wait_for_message('gazebo/model_states', ModelStates)
                for i in range(len(model.name)):
                    # Move obstacles
                    if model.name[i] == 'moving_obstacles_round':
                        obstacle.model_name = 'moving_obstacles_round'
                        obstacle.pose = model.pose[i]
                        obstacle.twist = Twist()
                        obstacle.twist.angular.z = 0.5
                        self.pub_model.publish(obstacle)
                        # rospy.sleep must always be wrapped in try/except in case the sim resets while sleeping
                        try:
                            rospy.sleep(0.1)
                        except rospy.exceptions.ROSTimeMovedBackwardsException, e:
                            print("Ros error due to reset during sleep, disregard")

            # When training in the gate module
            while self.current_module == "gate":
                gate = ModelState()
                model = rospy.wait_for_message('gazebo/model_states', ModelStates)
                for i in range(len(model.name)):
                    if model.name[i] == 'gate_moving':
                        gate.model_name = 'gate_moving'
                        gate.pose = model.pose[i]
                        gate.twist = Twist()
                        gate.twist.linear.z = GATE_MOVE_SPEED * self.gate_dir

                        # Gate is moving down and close to bottom, wait "shut"
                        if (model.pose[i].position.z < 0.3 and self.gate_dir == -1):
                            self.gate_dir = 1
                            gate.twist.linear.z = 0
                            gate.pose.position.z = 0.25
                            self.pub_model.publish(gate)
                            try:
                                rospy.sleep(GATE_WAIT_TIME)
                            except rospy.exceptions.ROSTimeMovedBackwardsException, e:
                                print("Ros error due to reset during sleep, disregard")
                            continue

                        # Gate is moving up and close to top, wait "open"
                        if (model.pose[i].position.z > 1 and self.gate_dir == 1):
                            self.gate_dir = -1
                            gate.twist.linear.z = 0
                            gate.pose.position.z = 1
                            self.pub_model.publish(gate)
                            try:
                                rospy.sleep(GATE_WAIT_TIME)
                            except rospy.exceptions.ROSTimeMovedBackwardsException, e:
                                print("Ros error due to reset during sleep, disregard")
                            continue

                        
                        self.pub_model.publish(gate)
                        try:
                            rospy.sleep(0.1)
                        except rospy.exceptions.ROSTimeMovedBackwardsException, e:
                            print("Ros error due to reset during sleep, disregard")
                


def main():
    rospy.init_node('moving_obstacle')
    moving = Moving()

if __name__ == '__main__':
    main()
