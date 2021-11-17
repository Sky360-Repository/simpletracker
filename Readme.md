# Overview

* Stage 1: Process videos looking for moving objects and generate annotation files with boundingboxes (tracks)
* Stage 2: Human associates tracks with labels
* Stage 3: Train NN

## Processing Stage 1 (SimpleTracker)

The purpose of this program is to capture or process 'interesting' videos and process them for input into Stage 2.

It can capture live images from an RSTP camera or a direct attached one and then generate videos, images and annotation files sutioable for input to Stage 2 and ultimately Stage 3 for training a Neural Net.

## Install

We recommend installing a conda environment and then running:

```./install.sh <location of conda> simpletracker```

## Running

You can run a test which will create an outputvideo.mp4 with bounding boxes rendered into the video of the given input video.

```
PYTHONPATH="${PYTHONPATH}:." \
python uap_tracker/main.py -f [input.mkv|input.mp4]
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

* SKY360_CONTROLLER Determines the input source type: \[Camera, Video\] 
* SKY360_VISUALIZER Determines the display output to screen: [two_by_two, simple, none] 
* SKY360_DETECTION_MODE Determines the method used as basis for object detection: [background_subtraction, optical_flow] 
* SKY360_OUTPUT_FORMAT Determines the MultipleObjectTracking or SingleObjectTracking file output mode: [mot_stf, sot_stf] 

Other settings can be found in the [settings.toml](https://github.com/Sky360-Repository/simpletracker/blob/master/settings.toml) file. Environemnt variables override settings.toml ones.

## Output

The output for the STF formatter is in this directory structure

```
 ./processed/                          # Processed videos
 ./stf/<video_name>_<section_id>/
   annotations.json                    # Annotations file
   video.mp4                           # Original video transcoded to mp4 and split for this section
   images/       
     <frame_id:06>.{image_name>.jpg    # Images generated during detection such as 'original', 'grey', 'background_subtraction'
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
