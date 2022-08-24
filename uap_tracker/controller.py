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

import cv2
import sys
import datetime
from datetime import timedelta
from uap_tracker.frame_processor import FrameProcessor
from uap_tracker.dense_optical_flow import DenseOpticalFlow
from uap_tracker.background_subtractor_factory import BackgroundSubtractorFactory

class Controller():

    def __init__(self):
        pass

    def run(self):
        pass

class CameraStreamController(Controller):

    def __init__(self, camera, video_tracker, minute_interval=10):
        super().__init__()

        self.camera = camera
        self.video_tracker = video_tracker
        self.minute_interval = minute_interval
        self.running = False

    def run(self):
        print("Running Camera")
        success, init_frame = self.camera.read()
        if not success:
            print(f"Could not open camera video stream")
            sys.exit()

        self.running = True

        while self.running:
            iteration_period = timedelta(minutes=self.minute_interval)
            self.process_iteration((datetime.datetime.now(
            ) + iteration_period), init_frame)

    def process_iteration(self, iteration_period, init_frame):

        frame_count = 0
        fps = 0
        background_subtractor = BackgroundSubtractorFactory.Select(enable_cuda=self.video_tracker.settings['enable_cuda'], sensitivity=self.video_tracker.settings['detection_sensitivity'])

        dense_optical_flow = None
        if self.video_tracker.settings['calculate_optical_flow']:
            dense_optical_flow = DenseOpticalFlow.Select(enable_cuda=self.video_tracker.settings['enable_cuda'], width=480, height=480)

        with FrameProcessor.Select(
            settings=self.video_tracker.settings,
            dense_optical_flow=dense_optical_flow,
            background_subtractor=background_subtractor) as processor:

            # Mike: Initialise the tracker and processor for this iteration
            self.video_tracker.initialise(processor, init_frame)

            while True:

                timer = cv2.getTickCount()
                success, frame = self.camera.read()
                if success:
                    self.video_tracker.process_frame(processor, frame, frame_count, fps)
                    # Calculate Frames per second (FPS)
                    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                    frame_count += 1

                if datetime.datetime.now() >= iteration_period:
                    # print(f"Iteration complete break")
                    if not self.video_tracker.is_tracking:
                        # print(f"Breaking loop")
                        self.video_tracker.finalise()
                        break

                if cv2.waitKey(1) == 27:  # Escape
                    print(
                        f"Escape keypress detected, exit even if we are tracking: {self.video_tracker.is_tracking}")
                    self.video_tracker.finalise()
                    self.running = False
                    break

class VideoPlaybackController(Controller):

    def __init__(self, capture, video_tracker):
        super().__init__()

        self.capture = capture
        self.video_tracker = video_tracker

    def run(self):

        success, init_frame = self.capture.read()
        if not success:
            print(f"Could not open video stream")
            sys.exit()

        frame_count = 0
        fps = 0
        background_subtractor = BackgroundSubtractorFactory.Select(enable_cuda=self.video_tracker.settings['enable_cuda'], sensitivity=self.video_tracker.settings['detection_sensitivity'])

        dense_optical_flow = None
        if self.video_tracker.settings['calculate_optical_flow']:
            dense_optical_flow = DenseOpticalFlow.Select(enable_cuda=self.video_tracker.settings['enable_cuda'], width=480, height=480)

        with FrameProcessor.Select(
            settings=self.video_tracker.settings,
            dense_optical_flow=dense_optical_flow,
            background_subtractor=background_subtractor) as processor:

            # Mike: Initialise the tracker and processor
            self.video_tracker.initialise(processor, init_frame)

            while cv2.waitKey(1) != 27:  # Escape
                success, frame = self.capture.read()
                if success:
                    timer = cv2.getTickCount()
                    self.video_tracker.process_frame(processor, frame, frame_count, fps)
                    # Calculate Frames per second (FPS)
                    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                    frame_count += 1
                else:
                    break

            self.video_tracker.finalise()