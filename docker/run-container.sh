#!/bin/bash

sudo docker run \
      -it \
      --runtime=nvidia \
      --gpus all \
      --privileged \
      -v /home/$USER/uap/simpletracker/:/home/simpletracker \
      opencv-cuda