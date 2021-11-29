#!/bin/bash

cd ~/simpletracker
git checkout nano_optimisation

mv ./settings.toml ./settings.toml.bak
mv ./settings.mike.toml ./settings.toml

sudo docker run \
      --runtime nvidia \
      -it --rm \
      --name simpletracker \
      --security-opt seccomp=unconfined \
      --net=host \
      -e DISPLAY=$DISPLAY \
      -v /home/$USER/simpletracker:/home/simpletracker \
      nvcr.io/nvidia/l4t-ml:r32.6.1-py3