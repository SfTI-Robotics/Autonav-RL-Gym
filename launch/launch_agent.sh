#!/bin/bash
# to be run from the root dir of this repo
# (should just run the bash script in root instead of running this directly)

DIR=$(pwd)
cd ../..
source ./devel/setup.bash

if [ $2 == "nav" ]; then
	echo $DIR
	export TURTLEBOT3_MODEL=burger
	roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$DIR/launch/nav_map/map.yaml &
	roslaunch project run_obstacles_nav.launch
else
	roslaunch project launch_agent.launch agent:=$2 run_type:=$1 load_ep:=$3
fi

while true
do
	:
done
