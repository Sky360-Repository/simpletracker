import cv2
import sys
import numpy as np
from uap_tracker.frame_processor import FrameProcessor
from uap_tracker.dense_optical_flow import DenseOpticalFlow
from uap_tracker.background_subtractor_factory import BackgroundSubtractorFactory

class VideoPlaybackController():

    def __init__(self, capture, video_tracker, enable_cuda=False):

        self.capture = capture
        self.video_tracker = video_tracker
        self.enable_cuda = enable_cuda

    def run(self):

        if not self.capture.isOpened():
            print(f"Could not open video stream")
            sys.exit()

        frame_count = 0
        fps = 0
        frame = np.empty((1024, 1024, 3),np.uint8)
        frame_grey = np.empty((1024, 1024, 3),np.uint8)
        frame_masked_background = np.empty((1024, 1024, 3),np.uint8)
        keypoints = []
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

            while cv2.waitKey(1) != 27:  # Escape
                success, frame = self.capture.read()
                if success:
                    timer = cv2.getTickCount()
                    self.video_tracker.process_frame(processor, frame, frame_grey, frame_masked_background, keypoints, frame_count, fps)
                    # Calculate Frames per second (FPS)
                    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                    frame_count += 1
                else:
                    break

            self.video_tracker.finalise()
