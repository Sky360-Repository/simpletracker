# usage: python uap_tracker/stage1.py -i <inputdir> -o <outputdirectory>
# e.g. python uap_tracker/stage1.py -i videos/samples/ -o videos/sp3

import os
import getopt
import sys
import cv2
from uap_tracker.video_tracker import VideoTracker
from uap_tracker.tracker_listener_dev import TrackerListenerDev
from uap_tracker.tracker_listener_stf import TrackerListenerMOTStf
from config import settings

USAGE = 'python uap_tracker/stage1.py\n settings are handled in the setttings.toml file or overridden in the ENV'

def _setup_tracker(video):
    return VideoTracker(video,detection_sensitivity=settings.VideoTracker.sensitivity, mask_pct=settings.VideoTracker.mask_pct)

def _setup_listener(video, full_path, root_name):
    formatters={
        'dev':TrackerListenerDev,
        'mot_stf':TrackerListenerMOTStf,
    }
    formatter_clz = formatters[settings.format]
    return formatter_clz(video, full_path, root_name, settings.output_dir)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", [])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
    
    print('Settings are ', settings.as_dict())

    if not os.path.isdir(settings.output_dir):
        os.mkdir(settings.output_dir)

    for filename in os.listdir(settings.input_dir):

        base = os.path.basename(filename)
        root_name = os.path.splitext(base)[0]

        full_path = os.path.join(settings.input_dir,filename)

        print(f"Opening {full_path}")
        video = cv2.VideoCapture(full_path)

        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        video_tracker=_setup_tracker(video)

        listener = _setup_listener(video, full_path, root_name)

        video_tracker.listen(listener)
        video_tracker.detect_and_track()

if __name__ == '__main__':
    main(sys.argv[1:])
