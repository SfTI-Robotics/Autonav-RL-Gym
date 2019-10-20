#!/bin/bash

#$1 train or test

gnome-terminal -e "./launch/launch_env.sh $1 $2"
sleep 4
gnome-terminal -e "./launch/launch_agent.sh $1 $2"
