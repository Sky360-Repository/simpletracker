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
from uap_tracker.frame_processor import FrameProcessor
from uap_tracker.dense_optical_flow import DenseOpticalFlow
from uap_tracker.background_subtractor_factory import BackgroundSubtractorFactory

class VideoPlaybackController():

    def __init__(self, capture, video_tracker, enable_cuda=False):

        self.capture = capture
        self.video_tracker = video_tracker
        self.enable_cuda = enable_cuda

    def run(self):

        success, init_frame = self.capture.read()
        if not success:
            print(f"Could not open video stream")
            sys.exit()

        frame_count = 0
        fps = 0
        background_subtractor = BackgroundSubtractorFactory.Select(enable_cuda=self.enable_cuda, sensitivity=self.video_tracker.detection_sensitivity)

        dense_optical_flow = None
        if self.video_tracker.calculate_optical_flow:
            dense_optical_flow = DenseOpticalFlow.Select(enable_cuda=self.enable_cuda, width=480, height=480)

        with FrameProcessor.Select(
                enable_cuda=self.enable_cuda,
                dense_optical_flow=dense_optical_flow,
                background_subtractor=background_subtractor,
                resize_frame=self.video_tracker.resize_frame,
                resize_dim=self.video_tracker.resize_dim,
                noise_reduction=self.video_tracker.noise_reduction,
                mask_pct=self.video_tracker.mask_pct,
                detection_mode=self.video_tracker.detection_mode,
                detection_sensitivity=self.video_tracker.detection_sensitivity) as processor:

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
