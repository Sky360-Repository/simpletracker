import cv2
import sys
import uap_tracker.utils as utils
from uap_tracker.video_tracker import VideoTracker

class VideoPlaybackController():

    def __init__(self, capture, visualiser=None, events=None, output_file=""):

        self.capture = capture
        self.visualiser = visualiser
        self.events = events
        self.output_file = output_file
        self.record = len(output_file) > 0
        self.video_tracker = None
        self.writer = None
        self.max_display_dim = 1080

    def run(self, detection_sensitivity=2, blur=False, normalise_video=False, mask_pct=92):

        if not self.capture.isOpened():
            print(f"Could not open video stream")
            sys.exit()

        source_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Open output video
        if self.record:
            self.writer = utils.get_writer(self.output_file, source_width, source_height)

        self.video_tracker = VideoTracker(self.visualiser, self.events, detection_sensitivity, mask_pct)

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
                if processed_frame.shape[0] > self.max_display_dim or processed_frame.shape[1] > self.max_display_dim:
                    # MG: scale the image to something that is of a reasonable viewing size
                    frame_scaled = utils.scale_image(processed_frame, self.max_display_dim)
                    cv2.imshow("Tracking", frame_scaled)
                else:
                    cv2.imshow("Tracking", processed_frame)

                if self.writer is not None:
                    self.writer.write(processed_frame)

                frame_count += 1
            else:
                break

        if self.writer is not None:
            self.writer.release()

        self.video_tracker.finalise()