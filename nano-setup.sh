#!/bin/bash

sudo apt update
sudo apt upgrade

sudo apt install gedit
sudo apt install python3-pip

pip3 install dynaconf

python3 -c 'import cv2; print(cv2.__version__)'

cd ~
mkdir Sky360-Repository
cd Sky360-Repository
git clone https://github.com/Sky360-Repository/simpletracker.git