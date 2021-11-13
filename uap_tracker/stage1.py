# usage: python uap_tracker/stage1.py -i <inputdir> -o <outputdirectory>
# e.g. python uap_tracker/stage1.py -i videos/samples/ -o videos/sp3

import os
import getopt
import sys
import cv2

from uap_tracker.default_visualiser import DefaultVisualiser
from uap_tracker.event_publisher import EventPublisher
from uap_tracker.simple_visualiser import SimpleVisualiser
from uap_tracker.tracker_listener_no_op import TrackerListenerNoOp
from uap_tracker.two_by_two_visualiser import TwoByTwoVisualiser
from uap_tracker.video_playback_controller import VideoPlaybackController
from uap_tracker.tracker_listener_dev import TrackerListenerDev
from uap_tracker.tracker_listener_stf import TrackerListenerMOTStf, TrackerListenerSOTStf
from config import settings

USAGE = 'python uap_tracker/stage1.py\n settings are handled in the setttings.toml file or overridden in the ENV'

def _setup_controller(video, events):
    visualizers={
        'default': DefaultVisualiser,
        'simple': SimpleVisualiser,
        'two_by_two': TwoByTwoVisualiser
    }
    visualizer_setting = settings.get('visualizer', 'default')
    visualizer_clz=visualizers[visualizer_setting]
    return VideoPlaybackController(video, visualiser=visualizer_clz(), events=events)

def _setup_listener(video, full_path, root_name):
    formatters={
        'dev':TrackerListenerDev,
        'noop': TrackerListenerNoOp,
        'mot_stf':TrackerListenerMOTStf,
        'sot_stf':TrackerListenerSOTStf
    }
    print(f"Initilaizing {settings.format}")
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

        events = EventPublisher()
        events.listen(_setup_listener(video, full_path, root_name))

        controller = _setup_controller(video, events)
        controller.run(detection_sensitivity=settings.VideoTracker.sensitivity, mask_pct=settings.VideoTracker.mask_pct)

if __name__ == '__main__':
    main(sys.argv[1:])
