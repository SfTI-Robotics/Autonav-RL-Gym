# Setup

This project requires ROS (Kinetic release), Gazebo (Ver 7), and catkin. For best compatibility use Ubutuntu 16.04.

1.  Create a catkin workspace:

```
$ source /opt/ros/kinetic/setup.bash
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
```
(Taken from http://wiki.ros.org/catkin/Tutorials/create_a_workspace)

2.  Install the required ROBOTIS Turtlebot 3 packages:

```
$ sudo apt-get install ros-kinetic-joy ros-kinetic-teleop-twist-joy ros-kinetic-teleop-twist-keyboard ros-kinetic-laser-proc ros-kinetic-rgbd-launch ros-kinetic-depthimage-to-laserscan ros-kinetic-rosserial-arduino ros-kinetic-rosserial-python ros-kinetic-rosserial-server ros-kinetic-rosserial-client ros-kinetic-rosserial-msgs ros-kinetic-amcl ros-kinetic-map-server ros-kinetic-move-base ros-kinetic-urdf ros-kinetic-xacro ros-kinetic-compressed-image-transport ros-kinetic-rqt-image-view ros-kinetic-gmapping ros-kinetic-navigation ros-kinetic-interactive-markers ros-kinetic-ros-control ros-kinetic-ros-controllers
```

(Taken from http://emanual.robotis.com/docs/en/platform/turtlebot3/pc_setup/#install-ubuntu-on-remote-pc)

3.  Add the required catkin packages to your catkin workspace:
```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
$ git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git
```

4.  Clone this repo to catkin_workspace
```
$ git clone https://github.com/SfTI-Robotics/Autonav-RL-Gym.git
```
5.  Run catkin_make:
```
$ cd ~/catkin_ws && catkin_make
```

6. Install the required python packages

```
pip install rospkg

```
For instructions on installing torch see here: https://pytorch.org/

# Running

To run the project, cd into the repo root (after running the above setup), and run the ./launch.sh file with the following arguments:

```
$ ./launch.sh [train/test] [ddpg/ppo/nav] [optional:episode number to load]
```

**Will start paused in Gazebo - click the play button in Gazebo at bottom left**

Arg 1:  whether to run the training or testing environment (training env is only designed for ppo and ddpg, not nav - turtlebot3 navigation package)

Arg 2:  Which agent to run (DDPG RL algorithm, PPO RL algorithm, "nav" Turtlebot3 Navigation package)

Arg 3 (optional): The number of the episode of a trained model to load. Must be present in the relevant src/saved_models folder, only supported for PPO and DDPG (as Turtlebot 3 navigation is not a learning model)

Occasionally Gazebo will error on launch, restarting usually fixes this.

Example: to run the PPO algorithm in the testing environment, loading episode 6000, run the command:
```
$ ./launch.sh test ppo 6000
```

# Information

This project was developed by Matthew Frost and Eugene Bulog, and supervised by Henry Williams, for a fourth year honors project at UoA. The goal of this project is to develop simulated training and testing environments for reinforcement learning in the area of mobile robotics autonomous navigation. Also included are two implementations of reinforcement learning algorithms. The simulated environments use the Gazebo and ROS platforms, and make use of the Turtlebot 3 robot models develepoed by Robotis. The DDPG implementation used is heavily adapted from one found at https://github.com/dranaju/project/tree/master/src. The Turtlebot 3 navigation package used to compare with our implementations was developed by Robotis.
