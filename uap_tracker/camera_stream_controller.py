import cv2
import sys
import datetime
from datetime import timedelta
import uap_tracker.utils as utils
from uap_tracker.video_tracker_new import VideoTrackerNew

class CameraStreamController():

    def __init__(self, camera_index=0, visualiser=None, events=None, output_file=""):

        self.camera_index = camera_index
        self.visualiser = visualiser
        self.events = events
        self.output_file = output_file
        self.record = len(output_file) > 0
        self.video_tracker = None
        self.writer = None
        self.max_display_dim = 1080
        self.minute_interval = 1
        self.execute = True

    def run(self, detection_sensitivity=2, blur=True, normalise_video=True, mask_pct=92):

        cameraCapture = cv2.VideoCapture(self.camera_index)

        success, _ = cameraCapture.read()
        if not success:
            print(f"Could not open video from camera {self.camera_index}")
            sys.exit()

        source_width = int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Open output video
        if self.record:
            self.writer = utils.get_writer(self.output_file, source_width, source_height)

        while self.execute:
            # print(f"execute iteration")
            iteration_period = timedelta(minutes=self.minute_interval)
            break_point = datetime.datetime.now() + iteration_period
            self.process_iteration(cameraCapture, break_point, detection_sensitivity, blur, normalise_video, mask_pct)

        if self.writer is not None:
            self.writer.release()

    def process_iteration(self, cameraCapture, iteration_period, detection_sensitivity, blur, normalise_video, mask_pct):

        self.video_tracker = VideoTrackerNew(self.visualiser, self.events, detection_sensitivity, mask_pct)

        # Read first frame.
        success, frame = cameraCapture.read()
        if success:
            self.video_tracker.initialise(frame, blur, normalise_video)

        for i in range(5):
            success, frame = cameraCapture.read()
            if success:
                self.video_tracker.initialise_background_subtraction(frame)

        self.video_tracker.initialise_trackers()

        frame_count = 0
        fps = 0
        while True:
            success, frame = cameraCapture.read()
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

            now = datetime.datetime.now()
            if now >= iteration_period:
                # print(f"Iteration complete break")
                if not self.video_tracker.is_tracking:
                    # print(f"Breaking loop")
                    self.video_tracker.finalise()
                    break

            if cv2.waitKey(1) == 27:  # Escape
                # print(f"Escape key, set execute to false and break even if we are tracking {self.video_tracker.is_tracking}")
                self.video_tracker.finalise()
                self.execute = False
                break
