#!/bin/bash

sudo apt update
sudo apt upgrade

#sudo apt install gedit
#sudo apt install python3-pip

#pip3 install dynaconf

#python3 -c 'import cv2; print(cv2.__version__)'

cd ~

git clone https://github.com/Sky360-Repository/simpletracker.git
git clone https://github.com/dusty-nv/jetson-inference.git

cd simpletracker
git checkout nano_optimisation

#sudo docker run \
#      --runtime nvidia \
#      -it --rm \
#      --name simpletracker \
#      --security-opt seccomp=unconfined \
#      --net=host \
#      -e DISPLAY=$DISPLAY \
#      -v /home/$USER/simpletracker:/app \
#      nvcr.io/nvidia/l4t-ml:r32.6.1-py3

#sudo docker exec -it nvc /bin/bash
#python3
