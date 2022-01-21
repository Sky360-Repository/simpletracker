#!/bin/bash

sudo apt install python3-pip3
sudo -H pip3 install -U jetson-statsf

while true; do
    read -p "Your Jetson device needs to be rebooted before you can run jtop. Reboot now? y to reboot, n to exit." yn
    case $yn in
        [Yy]* ) sudo shutdown -r now; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
