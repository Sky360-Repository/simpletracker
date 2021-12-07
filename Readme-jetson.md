##Installing jtop:

1. Launch the terminal
2. Change directory to simpletracker/scripts: ```cd ~/simpletracker/scripts```
3. Run the jtop installation script: ```./install-jtop.sh``` **NOTE:** The script needs to reboot your Jetson device for the installation to complete
4. After you have rebooted, you can access jtop by launching the terminal and typing ```jtop``` at the command prompt

More info on jtop can be found here: https://github.com/rbonghi/jetson_stats

##To get simpletracker running on the nano do the following:

1. Flash an SD card with Jetpack 4.6
2. Boot your nano using the SD card and complete the system setup
3. Launch the terminal
4. Update your package lists by typing: ```sudo apt update```
5. Upgrade your install by typing: ```sudo apt upgrade``` **NOTE:** This will take a while to run the first time as it needs to download and install all the system updates
   1.Select "Yes" to the docker restart prompt 
6. Clone the git repository using the following command: ```git clone https://github.com/Sky360-Repository/simpletracker.git```
7. Change directory to simpletracker/docker: ```cd ./simpletracker/docker```
8. Run the run file: ```./run.sh``` - **NOTE:** This will take a while to run the first time as it needs to download all the required container layers
9. Once you have a running command prompt type: ```cd /home/simpletracker/scripts```
10. Run the jetson run file: ```./jetson-run.sh```
    1. If you want output the results to a log file run the file using: ```./nano-jetson.sh >> timings_nano_cuda_perf_test.txt``` where ***timings_nano_cuda_perf_test.txt*** is the name of the file containing the output.   
11. When you are done type: ```exit``` to exit the container
12. Then type: ```exit``` again to exit the terminal
13. You can now use the file explorer to find your timings file which will be located in the simpletracker directory in your home folder
14. What I usually do is launch the browser and go to your email e.g. GMail
    1. Create a new email and attach the timings file to the email, it will then save that email as draft, you can now access it from anywhere. Alternatively upload to something like Google Drive or One Drive.

##Changing settings

The run.sh file located in the docker folder will remove the settings.toml file and replace it using the settings.mike.toml file. 
If you want setting changes to persist change both the settings.toml as well as the settings.mike.toml files. 
You can also create your own settings file and update the run.sh file in the docker folder to replace the settings.toml file with your settings file. 
1. Turning off the OpticalFlow  calcs by setting the ```calculate_optical_flow=``` to false (all lowercase)

##Enabling CUDA support

CUDA support is managed by the type of controller you would like to use. There are 2 main controller options at present, 
namely the Video Playback Controller and the Camera Controller. Both of these have CUDA support however this is still 
under active development. 

The main driver for which controller is used is driven by the *controller* setting in the setting.toml file. If you do 
not require CUDA support for e.g. video playback select the "video" option. If you are able to utilise CUDA then select the 
"video_cuda" option. There are further options but for these you need to review the comments in the settings.toml file.