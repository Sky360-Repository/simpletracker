# Overview

* Stage 1: Process videos looking for moving objects and generate annotation files with boundingboxes (tracks)
* Stage 2: Kinematics analysis by creating regression splines with increasing order when error rate exceeds threshold
* Stage 3: Human associates tracks with labels
* Stage 4: Crop slices from boundingboxes per frame
* Stage 5: Train NN on the slices

## Processing Stage 1 (SimpleTracker)

The purpose of this program is to capture or process 'interesting' videos and process them for input into Stage 2.

It can capture live images from an RSTP camera or a direct attached one and then generate videos, images and annotation files sutioable for input to Stage 2 and ultimately Stage 3 for training a Neural Net.

## How does SimpleTracker work?

Most parts of Simple Tracker is driven by configuration, so are tweakable to an extent

* There are 2 controller types that form the main entry points into SimpleTracker, namely the CameraController or the VideoController. One deals with a camera feed and the other with a video file.
* Both produce a sequence of frames which are then pushed throught a processing pipeline to determine if there are any trackable targets
* We will go into the processing pipeline a little further below as to what that means:
* Step 1: Mask is applied
* Step 2: Frame is resized
* Step 3: Frame is converted to Grey scale
* Step 4: A GaussianBlur is applied to reduce noise
* Step 5: Background subtraction is applied (CPU defaults to KNN, GPU defaults to MOG2)
* Step 6: Blobs are detected and keypoints extracted
* Step 7: Dense Optical Flow is run
* Step 8: Trackers (CSRT) are initialised using keypoints who are then tracked accross frames.
* Step 9: We use a Kalman Filter to try and predict the trajectory of the target being tracked
* Step 10: Events are raised, these mainly feed into the SimpleTracker listners which in turn will produce video files and other metadata files containing validated targets that were tracked. See the Output section below.
* Step 11: Visualisers are updated

Step 1 - 7 can be performed on both the CPU or GPU, however from Step 7 onwards its CPU only

## Install

We recommend using VSCode and runnning the application using the DevContainer. From the VSCode terminal type ```./run.sh`` to launch the application. Using the Dev Container there is no need to install dependencies as they are all in the container.

## Running

You can run a test which will create an outputvideo.mp4 with bounding boxes rendered into the video of the given input video.

Use the run file to launch the application
```
./run.sh
```

The full stage 1 capturing froma camera can be run using the following command:

```
PYTHONPATH="${PYTHONPATH}:." \
SKY360_CONTROLLER=Camera \
SKY360_VISUALIZER=two_by_two \
SKY360_DETECTION_MODE=background_subtraction \
SKY360_FORMAT=mot_stf \
python uap_tracker/main.py
```

Where:

* SKY360_CONTROLLER Determines the input source type: \[camera, video\] 
* SKY360_VISUALIZER Determines the display output to screen: [two_by_two, simple, none] 
* SKY360_DETECTION_MODE Determines the method used as basis for object detection: [background_subtraction]
* SKY360_OUTPUT_FORMAT Determines the MultipleObjectTracking or SingleObjectTracking file output mode: [mot_stf, sot_stf, none] 

Other settings can be found in the [settings.toml](https://github.com/Sky360-Repository/simpletracker/blob/master/settings.toml) file. Environemnt variables override settings.toml ones.

## Output

The output for the STF formatter is in this directory structure

```
 ./processed/                          # Processed videos
 ./stf/<video_name>_<section_id>/
   annotations.json                    # Annotations file
   video.mp4                           # Original video transcoded to mp4 and split for this section
   images/       
   training/                           # This will output traning images in incremnents of 32 pixels in size
     <frame_id:06>.{image_name>.jpg    # Images generated during detection such as 'original', 'grey', 'background_subtraction', 'optical_flow'
```

```
{
  "track_labels": {
    "1": "unknown"                     # track_id 1 has the label 'unknown'
  },
  "frames": [
    {
      "frame": 6,
      "annotations": {
        "bbox": [
          873,
          1222,
          49,
          49
        ],
        "track_id": 1
      }
    }...
```
