import cv2
import sys
import datetime
from datetime import timedelta


class CameraStreamControllerCuda():

    def __init__(self, camera, video_tracker, minute_interval=10):

        self.camera = camera
        self.video_tracker = video_tracker
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
        while True:
            timer = cv2.getTickCount()
            success, frame = self.camera.read()
            if success:

                self.video_tracker.process_frame_cuda(
                    frame, frame_count, fps)

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
