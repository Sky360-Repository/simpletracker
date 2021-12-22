import cv2
import sys
import datetime
from datetime import timedelta
import numpy as np
from uap_tracker.frame_processor import FrameProcessor
from uap_tracker.dense_optical_flow import DenseOpticalFlow
from uap_tracker.background_subtractor_factory import BackgroundSubtractorFactory

class CameraStreamController():

    def __init__(self, camera, video_tracker, enable_cuda=False, minute_interval=10):

        self.camera = camera
        self.video_tracker = video_tracker
        self.enable_cuda = enable_cuda
        self.minute_interval = minute_interval
        self.running = False

    def run(self):
        print("Running Camera")
        success, _ = self.camera.read()
        if not success:
            print(f"Could not open camera video stream")
            sys.exit()

        self.running = True

        while self.running:
            iteration_period = timedelta(minutes=self.minute_interval)
            self.process_iteration((datetime.datetime.now(
            ) + iteration_period))

    def process_iteration(self, iteration_period):

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
                noise_reduction=self.video_tracker.noise_reduction,
                mask_pct=self.video_tracker.mask_pct,
                detection_mode=self.video_tracker.detection_mode,
                detection_sensitivity=self.video_tracker.detection_sensitivity) as processor:

            while True:

                timer = cv2.getTickCount()
                success, frame = self.camera.read()
                if success:
                    self.video_tracker.process_frame(processor, frame, frame_grey, frame_masked_background, keypoints, frame_count, fps)
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
