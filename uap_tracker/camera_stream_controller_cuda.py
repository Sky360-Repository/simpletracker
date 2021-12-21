import cv2
import sys
import datetime
from datetime import timedelta
from uap_tracker.camera_stream_controller import CameraStreamController

class CameraStreamControllerCuda(CameraStreamController):

    def process_iteration(self, iteration_period):

        frame_count = 0
        fps = 0
        frame = np.empty((1024, 1024, 3),np.uint8)
        frame_grey = np.empty((1024, 1024, 3),np.uint8)
        frame_masked_background = np.empty((1024, 1024, 3),np.uint8)
        keypoints = []

        with FrameProcessor.GPU(
                resize_frame=self.video_tracker.resize_frame,
                noise_reduction=self.video_tracker.noise_reduction,
                mask_pct=self.video_tracker.mask_pct,
                detection_mode=self.video_tracker.detection_mode,
                detection_sensitivity=self.video_tracker.detection_sensitivity,
                blur_radius=self.video_tracker.blur_radius) as processor:

            while True:

                timer = cv2.getTickCount()
                success, frame = self.camera.read()
                if success:
                    self.video_tracker.process_frame(processor, frame, frame_grey, frame_masked_background, keypoints, frame_count, fps, stream=cv2.cuda.Stream_Null())
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
