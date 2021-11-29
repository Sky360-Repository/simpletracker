#!/bin/bash

# the purpose of this file is to run from within the docker container. We will eventually build a docker file that
# contains all of these things but at the moment I am testing using this mechanism.

pip3 install dynaconf

python3 -c 'import cv2; print(cv2.__version__)'

./run.sh