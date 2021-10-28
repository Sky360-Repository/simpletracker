import cv2
import sys
import numpy as np
import uap_tracker.utils as utils
from uap_tracker.tracker import Tracker
from uap_tracker.background_subtractor_factory import BackgroundSubtractorFactory

#
# Tracks multiple objects in a video
#
class VideoTracker():

    def __init__(self, video, detection_sensitivity=2):
        # print(f'VideoTracker called {video}')
        self.video = video
        self.detection_sensitivity = detection_sensitivity #Options are 1, 2, 3 TODO: Create enum for this
        self.total_trackers_finished = 0
        self.total_trackers_started = 0
        self.live_trackers = []
        self.listeners = []
        self.font_size = 8
        self.font_colour = (50, 170, 50)
        self.max_display_dim = 1080

    def listen(self, listener):
        self.listeners.append(listener)

    @property
    def is_tracking(self):
        return len(self.live_trackers) > 0

    def create_trackers_from_keypoints(self, tracker_type, key_points, frame, frame_hsv):
        for kp in key_points:
            bbox = utils.kp_to_bbox(kp)
            # print(bbox)

            # Initialize tracker with first frame and bounding box
            if not utils.is_bbox_being_tracked(self.live_trackers, bbox):
                self.create_and_add_tracker(tracker_type, frame, frame_hsv, bbox)

    def create_and_add_tracker(self, tracker_type, frame, frame_hsv, bbox):
        if not bbox:
            raise Exception("null bbox")

        self.total_trackers_started += 1

        tracker = Tracker(self.total_trackers_started, tracker_type, frame, frame_hsv, bbox, self.font_size, self.font_colour)
        tracker.update(frame, frame_hsv)
        self.live_trackers.append(tracker)

    def update_trackers(self, tracker_type, key_points, frame, frame_hsv):

        # cache kp -> bbox mapping for removing failed trackers
        kp_bbox_map = {}
        for kp in key_points:
            bbox = utils.kp_to_bbox(kp)
            kp_bbox_map[kp] = bbox

        failed_trackers = []
        for idx, tracker in enumerate(self.live_trackers):

            # Update tracker
            ok, bbox = tracker.update(frame, frame_hsv)
            if not ok:
                # Tracking failure
                failed_trackers.append(tracker)

            # Try to match the new detections with this tracker
            for idx, kp in enumerate(key_points):
                if kp in kp_bbox_map:
                    overlap = utils.bbox_overlap(bbox, kp_bbox_map[kp])
                    # print(f'Overlap: {overlap}; bbox:{bbox}, new_bbox:{new_bbox}')
                    if overlap > 0.2:
                        del (kp_bbox_map[kp])

        # remove failed trackers from live tracking
        for tracker in failed_trackers:
            self.live_trackers.remove(tracker)
            self.total_trackers_finished += 1

        # Add new detections to live tracker
        max_trackers = 10
        for kp, new_bbox in kp_bbox_map.items():
            # Hit max trackers?
            if len(self.live_trackers) < max_trackers:
                if not utils.is_bbox_being_tracked(self.live_trackers, new_bbox):
                    self.create_and_add_tracker(tracker_type, frame, frame_hsv, new_bbox)

    def detect_and_track(self, record=False, two_by_two=True):

        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'DASIAMRPN']
        background_subtractor_types = ['KNN']

        tracker_type = tracker_types[7]
        background_subtractor_type = background_subtractor_types[0]

        source_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Open output video
        writer = None
        if record:
            writer = utils.get_writer("outputvideo.mp4", source_width, source_height) #  MWG: I don't like the idea of this being here, TODO Move this into a listener

        self.font_size = int(source_height / 1000.0)

        # Read first frame.
        ok, frame = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        frame_gray = utils.convert_to_gray(frame)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        background_subtractor = BackgroundSubtractorFactory.create(background_subtractor_type, self.detection_sensitivity)

        frame_output, frame_masked_background = utils.apply_background_subtraction(frame_gray, background_subtractor)

        for i in range(5):
            ok, frame = self.video.read()
            frame_gray = utils.convert_to_gray(frame)
            frame_output, frame_masked_background = utils.apply_background_subtraction(frame_gray, background_subtractor)

        key_points = utils.perform_blob_detection(frame_masked_background, self.detection_sensitivity)

        # Create Trackers
        self.create_trackers_from_keypoints(tracker_type, key_points, frame_output, frame_hsv)

        frame_count = 0
        while True:
            # Read a new frame
            ok, frame = self.video.read()
            if not ok:
                break

            # Start timer
            timer = cv2.getTickCount()

            # Copy the frame as we want to mark the original and use the copy for displaying tracking artifacts
            frame_output = frame.copy()
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.putText(frame, 'Original Frame (Sky360)', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)

            # MG: This needs to be done on an 8 bit gray scale image, the colour image is causing a detection cluster
            _, frame_masked_background = utils.apply_background_subtraction(frame_gray, background_subtractor)

            # Detect new objects of interest to pass to tracker
            key_points = utils.perform_blob_detection(frame_masked_background, self.detection_sensitivity)

            self.update_trackers(tracker_type, key_points, frame_output, frame_hsv)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            for listener in self.listeners:
                listener.trackers_updated_callback(frame, frame_gray, frame_masked_background, frame_count+1, self.live_trackers, fps)

            msg = f"Trackers: started:{self.total_trackers_started}, ended:{self.total_trackers_finished}, alive:{len(self.live_trackers)}  (Sky360)"
            print(msg)
            cv2.putText(frame_output, msg, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)
            cv2.putText(frame_output, f"FPS: {str(int(fps))} (Sky360)", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)

            if two_by_two:
                # Create a copy as we need to put text on it and also turn it into a 24 bit image
                frame_masked_background_copy = cv2.cvtColor(frame_masked_background.copy(), cv2.COLOR_GRAY2BGR)

                frame_masked_background_with_key_points = cv2.drawKeypoints(frame_masked_background_copy, key_points, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                msg = f"Detected {len(key_points)} Key Points (Sky360)"
                cv2.putText(frame_masked_background_with_key_points, msg, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)

                cv2.putText(frame_masked_background_copy, "Masked Background (Sky360)", (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_size, self.font_colour, 2)

                im_h1 = cv2.hconcat([frame, frame_output])
                im_h2 = cv2.hconcat([frame_masked_background_copy, frame_masked_background_with_key_points])

                frame_final = cv2.vconcat([im_h1, im_h2])
            else:
                frame_final = frame_output

            # Display result, resize it to a standard size
            if frame_final.shape[0] > self.max_display_dim or frame_final.shape[1] > self.max_display_dim:
                #  MG: scale the image to something that is of a reasonable viewing size but write the original to file
                frame_scaled = utils.scale_image(frame_final, self.max_display_dim)
                cv2.imshow("Tracking", frame_scaled)
            else:
                cv2.imshow("Tracking", frame_final)

            if writer is not None:
                writer.write(frame_final)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

            frame_count += 1

        if writer is not None:
            writer.release()

        for listener in self.listeners:
            listener.finish(self.total_trackers_started, self.total_trackers_finished)
