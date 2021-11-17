import cv2
import sys
import uap_tracker.utils as utils
from uap_tracker.video_tracker import VideoTracker

class VideoPlaybackController():

    def __init__(self, capture, video_tracker):

        self.capture = capture
        self.video_tracker = video_tracker
        self.max_display_dim = 1080

    def run(self, blur=False, normalise_video=False):

        if not self.capture.isOpened():
            print(f"Could not open video stream")
            sys.exit()

        # Read first frame.
        success, frame = self.capture.read()
        if success:
            self.video_tracker.initialise(frame, blur, normalise_video)

        for i in range(5):
            success, frame = self.capture.read()
            if success:
                self.video_tracker.initialise_background_subtraction(frame)

        self.video_tracker.initialise_trackers()

        frame_count = 0
        fps = 0
        while cv2.waitKey(1) != 27:  # Escape
            success, frame = self.capture.read()
            if success:

                timer = cv2.getTickCount()

                processed_frame = self.video_tracker.process_frame(frame, frame_count, fps)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                # Display result, resize it to a standard size
                utils.display_frame(processed_frame, self.max_display_dim)

                frame_count += 1
            else:
                break

        self.video_tracker.finalise()
