import cv2
import sys
import datetime
from datetime import timedelta
import uap_tracker.utils as utils
from uap_tracker.video_tracker_new import VideoTrackerNew

class CameraStreamController():

    def __init__(self, camera_index=0, visualiser=None, events=None, record=False):

        self.camera_index = camera_index
        self.visualiser = visualiser
        self.events = events
        self.record = record
        self.video_tracker = None
        self.max_display_dim = 1080
        self.minute_interval = 5
        self.running = False
        self.source_width = 0
        self.source_height = 0

    def run(self, detection_sensitivity=2, blur=True, normalise_video=True, mask_pct=92):

        cameraCapture = cv2.VideoCapture(self.camera_index)

        success, _ = cameraCapture.read()
        if not success:
            print(f"Could not open video from camera {self.camera_index}")
            sys.exit()

        self.running = True
        self.source_width = int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while self.running:
            # print(f"execute iteration")
            iteration_period = timedelta(minutes=self.minute_interval)
            self.process_iteration(cameraCapture, (datetime.datetime.now() + iteration_period), detection_sensitivity, blur, normalise_video, mask_pct)

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

        recording = False
        writer = None
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

                frame_count += 1

                # MG: Setup the writer to record as we are tracking an object
                if self.record:
                    if self.video_tracker.is_tracking:
                        if not recording:
                            output_file = "sky360-tracking-" + datetime.datetime.now().strftime("%d-%m-%YT%H:%M:%S:%f") + ".mp4"
                            print(f"Recording to {output_file} as we are tracking")
                            writer = utils.get_writer(output_file, self.source_width, self.source_height)
                            recording = True
                    else:
                        if recording:
                            recording = False
                            print(f"Stop recording as we are no longer tracking")
                            if writer is not None:
                                writer.release()
                                writer = None

                if writer is not None:
                    writer.write(processed_frame)

            if datetime.datetime.now() >= iteration_period:
                # print(f"Iteration complete break")
                if not self.video_tracker.is_tracking:
                    # print(f"Breaking loop")
                    self.video_tracker.finalise()
                    break

            if cv2.waitKey(1) == 27:  # Escape
                print(f"Escape key, set execute to false and break even if we are tracking {self.video_tracker.is_tracking}")
                self.video_tracker.finalise()
                self.running = False
                break

        # MG: A bit of a catch all for the writer here
        if writer is not None:
            writer.release()
