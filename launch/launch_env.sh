#!/bin/bash
# to be run from the root dir of this repo
# (should just run the bash script in root instead of running this directly)

cd ../..
source ./devel/setup.bash

if [ $2 == "nav" ]; then
	roslaunch project $1ing_env.launch drive_type:=twist
else
	roslaunch project $1ing_env.launch 
fi

while true
do
	:
done
