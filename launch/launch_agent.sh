#!/bin/bash
# to be run from the root dir of this repo
# (should just run the bash script in root instead of running this directly)

cd ../..
source ./devel/setup.bash
roslaunch project $1_agent.launch agent:=$2

while true
do
	:
done
