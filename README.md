


# TurtleBot DDPG

Using DDPG actor critic to predict continuous action values for the purpose of navigation.


## Libraries

[Pytorch]

pip install torch

## ROS 
- https://github.com/ROBOTIS-GIT/turtlebot3
- https://github.com/ROBOTIS-GIT/turtlebot3_msgs
- https://github.com/ROBOTIS-GIT/turtlebot3_simulations

```
cd ~/catkin_ws/src/
git clone {link_git}
cd ~/catkin_ws && catkin_make
```

## IMPORTANT

Before running, need to modify turtlebot plugins to support individual wheel drives

In: turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro.

replace:

```
  <gazebo>
    <plugin name="turtlebot3_burger_controller" filename="libgazebo_ros_diff_drive.so">
      <commandTopic>cmd_vel</commandTopic>
      <odometryTopic>odom</odometryTopic>
      <odometryFrame>odom</odometryFrame>
      <odometrySource>world</odometrySource>
      <publishOdomTF>true</publishOdomTF>
      <robotBaseFrame>base_footprint</robotBaseFrame>
      <publishWheelTF>false</publishWheelTF>
      <publishTf>true</publishTf>
      <publishWheelJointState>true</publishWheelJointState>
      <legacyMode>false</legacyMode>
      <updateRate>30</updateRate>
      <leftJoint>wheel_left_joint</leftJoint>
      <rightJoint>wheel_right_joint</rightJoint>
      <wheelSeparation>0.160</wheelSeparation>
      <wheelDiameter>0.066</wheelDiameter>
      <wheelAcceleration>1</wheelAcceleration>
      <wheelTorque>10</wheelTorque>
      <rosDebugLevel>na</rosDebugLevel>
    </plugin>
  </gazebo>
```

with:

```
  <gazebo>
    <plugin name="turtlebot3_burger_controller_l" filename="libgazebo_ros_diff_drive.so">
      <commandTopic>cmd_vel_l</commandTopic>
      <odometryTopic>odom</odometryTopic>
      <odometryFrame>odom</odometryFrame>
      <odometrySource>world</odometrySource>
      <publishOdomTF>true</publishOdomTF>
      <robotBaseFrame>base_footprint</robotBaseFrame>
      <publishWheelTF>false</publishWheelTF>
      <publishTf>true</publishTf>
      <publishWheelJointState>true</publishWheelJointState>
      <legacyMode>false</legacyMode>
      <updateRate>30</updateRate>
      <leftJoint>wheel_left_joint</leftJoint>
      <rightJoint>wheel_left_joint</rightJoint>
      <wheelSeparation>0</wheelSeparation>
      <wheelDiameter>0.066</wheelDiameter>
      <wheelAcceleration>1</wheelAcceleration>
      <wheelTorque>10</wheelTorque>
      <rosDebugLevel>na</rosDebugLevel>
    </plugin>
  </gazebo>

  <gazebo>
    <plugin name="turtlebot3_burger_controller_r" filename="libgazebo_ros_diff_drive.so">
      <commandTopic>cmd_vel_r</commandTopic>
      <odometryTopic>odom</odometryTopic>
      <odometryFrame>odom</odometryFrame>
      <odometrySource>world</odometrySource>
      <publishOdomTF>true</publishOdomTF>
      <robotBaseFrame>base_footprint</robotBaseFrame>
      <publishWheelTF>false</publishWheelTF>
      <publishTf>true</publishTf>
      <publishWheelJointState>true</publishWheelJointState>
      <legacyMode>false</legacyMode>
      <updateRate>30</updateRate>
      <leftJoint>wheel_right_joint</leftJoint>
      <rightJoint>wheel_right_joint</rightJoint>
      <wheelSeparation>0</wheelSeparation>
      <wheelDiameter>0.066</wheelDiameter>
      <wheelAcceleration>1</wheelAcceleration>
      <wheelTorque>10</wheelTorque>
      <rosDebugLevel>na</rosDebugLevel>
    </plugin>
  </gazebo>
```


## Set State

In: turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro.

```
<xacro:arg name="laser_visual" default="false"/>   # Visualization of LDS. If you want to see LDS, set to `true`
```
And
```
<scan>
  <horizontal>
    <samples>360</samples>            # The number of sample. Modify it to 24
    <resolution>1</resolution>
    <min_angle>0.0</min_angle>
    <max_angle>6.28319</max_angle>
  </horizontal>
</scan>
```

## Run Code

```
roslaunch turtlebot3_gazebo turtlebot3_stage_{number_of_stage}.launch
```
In another terminal run:
```
roslaunch project ddpg_stage_{number_of_stage}.launch
```

## Run Training Environment

SEE IMPORTANT SECTION ON CHANGING TURTLEBOT DEFINITION FIRST
Open two terminals, in both run (from catkin workspace root):
```
source ./devel/setup.bash
export TURTLEBOT3_MODEL=burger
```

In one terminal, run the environment:
```
roslaunch project training_env.launch
```
In the other, run the agent (either sac or dppg as the argument):
```
roslaunch project train_agent.launch agent:=ddpg
```


## Logging
Install logstash, elasticsearch, kibana

Run the logstash pipeline from the root of this repo before running the turtlebot agent:
```
/path/to/logstash -f ./RL-pipeline.conf
```
In separate terminals run elasticsearch and kibana (no arguments necessary)
Once kibana is running import the config data by following the import instructions here: https://support.logz.io/hc/en-us/articles/210207225-How-can-I-export-import-Dashboards-Searches-and-Visualizations-from-my-own-Kibana-
and importing the kibanaData.ndjson file

In kibana open up the "RL Dash" dashboard
Add a filter "path is [path to log file]" to filter it to a specific run of the agent





