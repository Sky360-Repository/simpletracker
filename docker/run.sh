#!/bin/bash

rm ../settings.toml
cp ../settings.mike.toml ../settings.toml

sudo docker run \
      --runtime nvidia \
      --name simpletracker \
      -it --rm --net=host \
      --security-opt seccomp=unconfined \
      -e DISPLAY=$DISPLAY \
      -v /tmp/.X11-unix/:/tmp/.X11-unix \
      -v /tmp/argus_socket:/tmp/argus_socket \
      -v /etc/enctune.conf:/etc/enctune.conf \
      -v /home/$USER/simpletracker:/home/simpletracker \
      nvcr.io/nvidia/l4t-ml:r32.6.1-py3