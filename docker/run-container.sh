#!/bin/bash

sudo docker run \
      -it \
      --runtime=nvidia \
      --gpus all \
      --privileged \
      -v /home/$USER/uap/simpletracker/:/app \
      -v /home/$USER/uap/rcnn:/rcnn \
      opencv-cuda