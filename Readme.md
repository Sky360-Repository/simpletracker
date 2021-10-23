# Historical Processing Stage 1 (SimpleTracker)

The purpose of this program is to take historical videos created with the Skyhub/Sky360 program and process them for input into Stage 2.

## Install

We recommend installing a conda environment and then running:

```conda create --name simpletracker --file requirements.txt```

You can also install with pip:

```pip install -r requirements.txt```

## Running

You can run a test which will create an outputvideo.mp4 with bounding boxes rendered into the video.

```PYTHONPATH=. python uap_tracker/detect_and_track.py [filename.mkv|filename.mp4]```

The full stage 1 can be run using the following command:

```PYTHONPATH=. python uap_tracker/stage1.py -i videos/samples/ -o videos/sp14 -f stf```

Where:

* -i is the directory containing the videos to be processed
* -o is the directory where you want the output to go
* -f is the output format

