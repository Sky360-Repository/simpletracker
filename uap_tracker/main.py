# Original work Copyright (c) 2022 Sky360
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# usage: python uap_tracker/stage1.py
# e.g. python uap_tracker/stage1.py

from datetime import datetime
import os
import getopt
import sys
import cv2
import shutil

from uap_tracker.event_publisher import EventPublisher
from uap_tracker.no_op_visualiser import NoOpVisualiser
from uap_tracker.simple_visualiser import SimpleVisualiser
from uap_tracker.two_by_two_optical_flow_visualiser import TwoByTwoOpticalFlowVisualiser
from uap_tracker.two_by_two_visualiser import TwoByTwoVisualiser
from uap_tracker.video_playback_controller import VideoPlaybackController
from uap_tracker.camera_stream_controller import CameraStreamController
from uap_tracker.tracker_listener_stf import TrackerListenerMOTStf, TrackerListenerSOTStf
from uap_tracker.video_frame_dumpers import OriginalFrameVideoWriter, GreyFrameVideoWriter, OpticalFlowFrameVideoWriter, AnnotatedFrameVideoWriter, MaskedBackgroundFrameVideoWriter
from config import settings
from uap_tracker.video_tracker import VideoTracker
from camera import get_camera
from video_formatter import VideoFormatter
import uap_tracker.utils as utils


USAGE = 'python uap_tracker/main.py\n settings are handled in the setttings.toml file or overridden in the ENV'

def _setup_controller(media, events, visualizer, detection_mode):
    controller_clz = _get_controller()

    video_tracker = VideoTracker(
        detection_mode,
        events,
        visualizer,
        detection_sensitivity=settings.VideoTracker.sensitivity,
        mask_pct=settings.VideoTracker.mask_pct,
        noise_reduction=settings.VideoTracker.get('noise_reduction', False),
        resize_frame=settings.VideoTracker.get('resize_frame', False),
        resize_dim=settings.VideoTracker.resize_dim,
        calculate_optical_flow=settings.VideoTracker.calculate_optical_flow,
        max_active_trackers=settings.VideoTracker.max_active_trackers,
    )

    return controller_clz(media, video_tracker, settings.enable_cuda)


def _get_visualizer(detection_mode):
    two_by_two_mode_visualizers = {
        'background_subtraction': TwoByTwoVisualiser,
        'optical_flow': TwoByTwoOpticalFlowVisualiser,
        'none': None
    }

    visualizers = {
        'none': None,
        'noop': NoOpVisualiser,
        'simple': SimpleVisualiser,
        'two_by_two': two_by_two_mode_visualizers[detection_mode]
    }
    visualizer_format = settings.Visualizer.get('format', None)
    visualizer_max_display_dim = settings.Visualizer.get(
        'max_display_dim', None)

    if not visualizer_format:
        print(
            f"Please set 'visualizer' in the config or use the SKY360_VISUALIZER env var: {visualizers.keys()}")
        sys.exit(1)

    visualizer_clz = visualizers[visualizer_format]

    if visualizer_clz:
        visualizer = visualizer_clz(
            visualizer_max_display_dim)
    else:
        visualizer = None
    print(f"Visualizer: {visualizer}")
    return visualizer


def _get_detection_mode():
    detection_mode = settings.VideoTracker.get(
        'detection_mode', None)

    detection_modes = ['background_subtraction', 'optical_flow', 'none']

    if not detection_mode:
        print(
            f"Please set detection_mode in the config or use the SKY360_DETECTION_MODE env var: {detection_modes}")
        sys.exit(1)
    else:
        print(f"Detection Mode: {detection_mode}")
    return detection_mode


def _get_controller():
    controllers = {
        'video': VideoPlaybackController,
        'camera': CameraStreamController,
    }
    controller_setting = settings.get('controller', None)

    if not controller_setting:
        print(
            f"Please set controller in the config or use the SKY360_CONTROLLER env var: {controllers.keys()}")
        sys.exit(1)

    controller_clz = controllers[controller_setting]
    return controller_clz


def _setup_listener(video, root_name, output_dir):
    formatters = {
        'none': None,
        'mot_stf': TrackerListenerMOTStf,
        'sot_stf': TrackerListenerSOTStf,
        'video': VideoFormatter,
    }
    print(f"Initilaizing {settings.output_format}")
    formatter_clz = formatters[settings.output_format]
    if formatter_clz:
        return formatter_clz(video, root_name, output_dir)


def _setup_dumpers(video, root_name, output_dir):
    dumpers = {
        'none': None,
        'dump_original': OriginalFrameVideoWriter,
        'dump_grey': GreyFrameVideoWriter,
        'dump_optical_flow': OpticalFlowFrameVideoWriter,
        'dump_annotated': AnnotatedFrameVideoWriter,
        'dump_masked_background': MaskedBackgroundFrameVideoWriter,
    }
    print(f"Dumpers {settings.frame_dumpers}")
    if settings.frame_dumpers == 'all':
        return [OriginalFrameVideoWriter(video, root_name, output_dir),
                GreyFrameVideoWriter(video, root_name, output_dir),
                OpticalFlowFrameVideoWriter(video, root_name, output_dir),
                AnnotatedFrameVideoWriter(video, root_name, output_dir),
                MaskedBackgroundFrameVideoWriter(video, root_name, output_dir)]
    else:
        dumper_clz = dumpers[settings.frame_dumpers]
        if dumper_clz:
            return [dumper_clz(video, root_name, output_dir)]


def main(argv):

    print(f"Open CV Version: {cv2.__version__}")

    if not utils.is_cv_version_supported():
        print(f"Unfortunately OpenCV v{cv2.__version__} is not supported, we support v4.1 and above.")
        sys.exit(1)

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

    #cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)

    output_dir = _create_output_dir()

    controller = _get_controller()

    detection_mode = _get_detection_mode()

    visualizer = _get_visualizer(detection_mode)

    # If a video was passed in on commandline, run that and ignore other sources
    if cmdline_filename:
        process_file(controller, visualizer, cmdline_filename,
                     output_dir, detection_mode)
    else:
        if controller == VideoPlaybackController:

            processed_dir = os.path.join(settings.input_dir, "processed")
            if not os.path.isdir(processed_dir):
                os.mkdir(processed_dir)

            sorted_files = os.listdir(settings.input_dir)
            sorted_files.sort()

            for filename in sorted_files:
                full_path = os.path.join(settings.input_dir, filename)
                process_file(controller, visualizer, full_path,
                             output_dir, detection_mode)
                processed_path = os.path.join(processed_dir, filename)
                shutil.move(full_path,processed_path)

        elif controller == CameraStreamController:
            camera = get_camera(settings.get('camera', {}))
            listener = _setup_listener(camera, 'capture', output_dir)
            dumpers = _setup_dumpers(camera, 'capture', output_dir)
            if dumpers is not None:
                _run(controller, [listener] + dumpers, visualizer, camera, detection_mode)
            else:
                _run(controller, [listener], visualizer, camera, detection_mode)


def _create_output_dir():
    if not os.path.isdir(settings.output_dir):
        os.mkdir(settings.output_dir)

    time_based_dir = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = os.path.join(settings.output_dir, time_based_dir)
    os.mkdir(output_dir)
    return output_dir


def process_file(controller, visualizer, full_path, output_dir, detection_mode):
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
    dumpers = _setup_dumpers(video, root_name, output_dir)
    if dumpers is not None:
        _run(controller, [listener] + dumpers, visualizer, video, detection_mode)
    else:
        _run(controller, [listener], visualizer, video, detection_mode)


def _run(controller, listeners, visualizer, media, detection_mode):

    events = EventPublisher()

    for listener in listeners:
        if listener:
            events.listen(listener)

    controller = _setup_controller(media, events, visualizer, detection_mode)
    controller.run()


if __name__ == '__main__':
    main(sys.argv[1:])
