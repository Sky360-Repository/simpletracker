import cv2
import sys


class VideoPlaybackController():

    def __init__(self, capture, video_tracker):

        self.capture = capture
        self.video_tracker = video_tracker

    def run(self):

        if not self.capture.isOpened():
            print(f"Could not open video stream")
            sys.exit()

        frame_count = 0
        fps = 0
        while cv2.waitKey(1) != 27:  # Escape
            success, frame = self.capture.read()
            if success:

                timer = cv2.getTickCount()

                self.video_tracker.process_frame(frame, frame_count, fps)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                frame_count += 1
            else:
                break

        self.video_tracker.finalise()
