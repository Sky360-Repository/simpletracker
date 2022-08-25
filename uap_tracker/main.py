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
from uap_tracker.visualizer import NoOpVisualiser, SimpleVisualiser, TwoByTwoVisualiser, TwoByTwoOpticalFlowVisualiser
from uap_tracker.controller import VideoPlaybackController, CameraStreamController
from uap_tracker.tracker_listener_stf import TrackerListenerMOTStf, TrackerListenerSOTStf
from uap_tracker.video_frame_dumpers import OriginalFrameVideoWriter, GreyFrameVideoWriter, OpticalFlowFrameVideoWriter, AnnotatedFrameVideoWriter, MaskedBackgroundFrameVideoWriter
from config import settings
from uap_tracker.video_tracker import VideoTracker
from camera import get_camera
from video_formatter import VideoFormatter
import uap_tracker.utils as utils
from app_settings import AppSettings


USAGE = 'python uap_tracker/main.py\n settings are handled in the setttings.toml file or overridden in the ENV'

def _setup_controller(media, events, visualizer, app_settings):
    controller_clz = _get_controller()

    video_tracker = VideoTracker(app_settings, events, visualizer)

    return controller_clz(media, video_tracker)


def _get_visualizer(app_settings):

    two_by_two_mode_visualizers = {
        'background_subtraction': TwoByTwoVisualiser,
        'optical_flow': TwoByTwoOpticalFlowVisualiser,
        'none': None
    }

    visualizers = {
        'none': None,
        'noop': NoOpVisualiser,
        'simple': SimpleVisualiser,
        'two_by_two': two_by_two_mode_visualizers[app_settings['detection_mode']]
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
        visualizer = visualizer_clz(visualizer_max_display_dim, 
        app_settings['font_size'], app_settings['font_thickness'])
    else:
        visualizer = None
    print(f"Visualizer: {visualizer}")
    return visualizer


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

    print(f"Open CV Version: {cv2.__version__}, CUDA support: {utils.is_cuda_supported()}")

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

    app_settings = AppSettings.Get(settings)
    AppSettings.Validate(app_settings)

    output_dir = _create_output_dir()

    controller = _get_controller()

    visualizer = _get_visualizer(app_settings)

    # If a video was passed in on commandline, run that and ignore other sources
    if cmdline_filename:
        process_file(controller, visualizer, cmdline_filename,
                     output_dir, app_settings)
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
                             output_dir, app_settings)
                processed_path = os.path.join(processed_dir, filename)
                shutil.move(full_path,processed_path)

        elif controller == CameraStreamController:
            camera = get_camera(settings.get('camera', {}))
            listener = _setup_listener(camera, 'capture', output_dir)
            dumpers = _setup_dumpers(camera, 'capture', output_dir)
            if dumpers is not None:
                _run(controller, [listener] + dumpers, visualizer, camera, app_settings)
            else:
                _run(controller, [listener], visualizer, camera, app_settings)


def _create_output_dir():
    if not os.path.isdir(settings.output_dir):
        os.mkdir(settings.output_dir)

    time_based_dir = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = os.path.join(settings.output_dir, time_based_dir)
    os.mkdir(output_dir)
    return output_dir


def process_file(controller, visualizer, full_path, output_dir, app_settings):
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
        _run(controller, [listener] + dumpers, visualizer, video, app_settings)
    else:
        _run(controller, [listener], visualizer, video, app_settings)


def _run(controller, listeners, visualizer, media, app_settings):

    events = EventPublisher()

    for listener in listeners:
        if listener:
            events.listen(listener)

    controller = _setup_controller(media, events, visualizer, app_settings)
    controller.run()


if __name__ == '__main__':
    main(sys.argv[1:])
