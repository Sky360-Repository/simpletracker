# usage: python uap_tracker/stage1.py
# e.g. python uap_tracker/stage1.py

from datetime import datetime
import os
import getopt
import sys
import cv2

from uap_tracker.default_visualiser import DefaultVisualiser
from uap_tracker.event_publisher import EventPublisher
from uap_tracker.simple_visualiser import SimpleVisualiser
from uap_tracker.two_by_two_optical_flow_visualiser import TwoByTwoOpticalFlowVisualiser
from uap_tracker.two_by_two_visualiser import TwoByTwoVisualiser
from uap_tracker.video_playback_controller import VideoPlaybackController
from uap_tracker.camera_stream_controller import CameraStreamController
from uap_tracker.tracker_listener_stf import TrackerListenerMOTStf, TrackerListenerSOTStf
from config import settings
from uap_tracker.video_tracker import VideoTracker
from camera import get_camera

USAGE = 'python uap_tracker/stage1.py\n settings are handled in the setttings.toml file or overridden in the ENV'


def _setup_controller(media, events):
    controller_clz = _get_controller()

    detection_mode = settings.get(
        'detection_mode', 'background_subtraction')

    two_by_two_mode_visualizers = {
        'background_subtraction': TwoByTwoVisualiser,
        'optical_flow': TwoByTwoOpticalFlowVisualiser
    }

    visualizers = {
        'none': None,
        'default': DefaultVisualiser,
        'simple': SimpleVisualiser,
        'two_by_two': two_by_two_mode_visualizers[detection_mode]
    }
    visualizer_setting = settings.get('visualizer', 'default')
    visualizer_clz = visualizers[visualizer_setting]

    visualizer = visualizer_clz() if visualizer_clz else None
    print(f"Visualizer: {visualizer}")
    video_tracker = VideoTracker(
        detection_mode,
        visualizer,
        events,
        detection_sensitivity=settings.VideoTracker.sensitivity,
        mask_pct=settings.VideoTracker.mask_pct
    )

    return controller_clz(media, video_tracker)


def _get_controller():
    controllers = {
        'video': VideoPlaybackController,
        'camera': CameraStreamController
    }
    controller_setting = settings.get('controller', 'Video')
    controller_clz = controllers[controller_setting]
    return controller_clz


def _setup_listener(video, root_name, output_dir):
    formatters = {
        'none': None,
        'mot_stf': TrackerListenerMOTStf,
        'sot_stf': TrackerListenerSOTStf
    }
    print(f"Initilaizing {settings.output_format}")
    formatter_clz = formatters[settings.output_format]
    if formatter_clz:
        return formatter_clz(video, root_name, output_dir)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hf:", [])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)

    cmdline_filename = None
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        if opt == '-f':
            cmdline_filename = arg
    print(f"cmdline_filename: {cmdline_filename}")
    print('Settings are ', settings.as_dict())

    output_dir = _create_output_dir()

    controller = _get_controller()

    if controller == VideoPlaybackController:

        if cmdline_filename:
            process_file(controller, cmdline_filename, output_dir)
        else:
            for filename in os.listdir(settings.input_dir):
                full_path = os.path.join(settings.input_dir, filename)
                process_file(controller, full_path, output_dir)
    elif controller == CameraStreamController:
        camera = get_camera(settings.get('camera', {}))
        listener = _setup_listener(camera, 'capture', output_dir)

        _run(controller, listener, camera)

def _create_output_dir():
    if not os.path.isdir(settings.output_dir):
        os.mkdir(settings.output_dir)

    time_based_dir = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    output_dir = os.path.join(settings.output_dir, time_based_dir)
    os.mkdir(output_dir)
    return output_dir


def process_file(controller, full_path, output_dir):
    base = os.path.basename(full_path)
    root_name = os.path.splitext(base)[0]

    print(f"Opening {full_path}")
    video = cv2.VideoCapture(full_path)
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    events = EventPublisher()
    listener = _setup_listener(video, root_name, output_dir)
    if listener:
        events.listen(listener)

    _run(controller, listener, video)


def _run(controller, listener, media):

    events = EventPublisher()

    if listener:
        events.listen(listener)

    controller = _setup_controller(media, events)
    controller.run(
        normalise_video=settings.VideoTracker.get('normalize', False)
    )


if __name__ == '__main__':
    main(sys.argv[1:])
