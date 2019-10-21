#!/bin/bash

#$1 train or test
#$2 agent type (ddpg, ppo, nav (for turtlebot3 navigation package)
#$3 load ep (optional)

gnome-terminal -e "./launch/launch_env.sh $1 $2"
sleep 4
gnome-terminal -e "./launch/launch_agent.sh $1 $2 $3"
