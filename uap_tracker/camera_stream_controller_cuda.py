import cv2
import sys
import datetime
from datetime import timedelta
from uap_tracker.camera_stream_controller import CameraStreamController

class CameraStreamControllerCuda(CameraStreamController):

    def process_iteration(self, iteration_period):

        frame_count = 0
        fps = 0
        while True:
            timer = cv2.getTickCount()
            success, frame = self.camera.read()
            if success:

                with FrameProcessor.GPU(frame, self.video_tracker.dof_cuda) as processor:
                    self.video_tracker.process_frame_cuda(processor, frame, frame_count, fps)

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
