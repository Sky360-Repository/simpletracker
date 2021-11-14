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
from uap_tracker.camera_stream_controller import CameraStreamController
from uap_tracker.tracker_listener_dev import TrackerListenerDev
from uap_tracker.tracker_listener_stf import TrackerListenerMOTStf, TrackerListenerSOTStf
from config import settings
from uap_tracker.video_tracker import VideoTracker

USAGE = 'python uap_tracker/stage1.py\n settings are handled in the setttings.toml file or overridden in the ENV'


def _setup_controller(media, events):
    controller_clz = get_controller()

    visualizers = {
        'default': DefaultVisualiser,
        'simple': SimpleVisualiser,
        'two_by_two': TwoByTwoVisualiser
    }
    visualizer_setting = settings.get('visualizer', 'default')
    visualizer_clz = visualizers[visualizer_setting]

    return controller_clz(media, visualiser=visualizer_clz(), events=events)


def get_controller():
    controllers = {
        'Video': VideoPlaybackController,
        'Camera': CameraStreamController
    }
    controller_setting = settings.get('controller', 'Video')
    controller_clz = controllers[controller_setting]
    return controller_clz


def _setup_listener(video, full_path, root_name):
    formatters = {
        'none': None,
        'dev': TrackerListenerDev,
        'noop': TrackerListenerNoOp,
        'mot_stf': TrackerListenerMOTStf,
        'sot_stf': TrackerListenerSOTStf
    }
    print(f"Initilaizing {settings.format}")
    formatter_clz = formatters[settings.format]
    if formatter_clz:
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

    controller = get_controller()

    if controller == VideoPlaybackController:

        for filename in os.listdir(settings.input_dir):

            base = os.path.basename(filename)
            root_name = os.path.splitext(base)[0]

            full_path = os.path.join(settings.input_dir, filename)

            print(f"Opening {full_path}")
            video = cv2.VideoCapture(full_path)
            # Exit if video not opened.
            if not video.isOpened():
                print("Could not open video")
                sys.exit()

            events = EventPublisher()
            listener = _setup_listener(video, full_path, root_name)
            if listener:
                events.listen(listener)

            _run(controller, listener, video)
    elif controller == CameraStreamController:
        full_path = os.path.join(settings.input_dir, 'streaming')
        camera = get_camera(settings.get('camera', {}))
        listener = _setup_listener(camera, full_path, 'capture')

        _run(controller, listener, camera)


def get_camera(config):
    camera_mode = config.get('mode', 'rtsp')
    camera_uri = config.get('camera_uri',
                            'rtsp://admin:sky360@192.168.1.108:554/live')
    print(f"Connecting to {camera_mode} camera at {camera_uri}")
    camera = None
    if camera_mode == 'ffmpeg':
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        camera = cv2.VideoCapture(
            camera_uri,
            cv2.CAP_FFMPEG
        )
    elif camera_mode == 'rtsp':
        camera = cv2.VideoCapture(
            camera_uri
        )
    elif camera_mode == 'local':
        for i in range(-1, 100):
            try:
                camera = cv2.VideoCapture(i)
                if camera:
                    break
            except cv2.error as e:
                print(e)
            except Exception as e:
                print(e)
    ##
    ## Apparently this works for firewire, but I couldn't get it to
    ## camera = cv2.VideoCapture(3, cv2.CAP_DC1394)

    if not camera:
        print(cv2.getBuildInformation())
        print(f"Unable to find camera using config: {config}")
        sys.exit(1)
    return camera


def _run(controller, listener, media):

    events = EventPublisher()

    if listener:
        events.listen(listener)

    controller = _setup_controller(media, events)
    controller.run(
        detection_sensitivity=settings.VideoTracker.sensitivity,
        mask_pct=settings.VideoTracker.mask_pct,
        normalise_video=settings.VideoTracker.get('normalize',False)
    )


if __name__ == '__main__':
    main(sys.argv[1:])
