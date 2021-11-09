import cv2
import numpy as np
import uap_tracker.utils as utils
from uap_tracker.tracker import Tracker
from uap_tracker.background_subtractor_factory import BackgroundSubtractorFactory

#
# Tracks multiple objects in a video
#
class VideoTrackerNew():

    DETECTION_SENSITIVITY_HIGH = 1
    DETECTION_SENSITIVITY_NORMAL = 2
    DETECTION_SENSITIVITY_LOW = 3

    def __init__(self, visualiser, events, detection_sensitivity=2, mask_pct=92):
        # print(f'VideoTracker called {video}')

        if detection_sensitivity < 1 or detection_sensitivity > 3:
            raise Exception(
                f"Unknown sensitivity option ({detection_sensitivity}). 1, 2 and 3 is supported not {detection_sensitivity}.")

        self.detection_sensitivity = detection_sensitivity
        self.total_trackers_finished = 0
        self.total_trackers_started = 0
        self.live_trackers = []
        self.events = events
        self.visualiser = visualiser
        self.font_size = 8
        self.font_colour = (50, 170, 50)
        self.normalised_w_h = (1920, 1080)
        self.blur_radius = 3
        self.max_active_trackers = 10
        self.mask_pct = mask_pct

        self.blur = False
        self.normalise_video = False
        self.tracker_type = None
        self.background_subtractor_type = None
        self.background_subtractor = None
        self.frame_output = None
        self.frame_masked_background = None

    @property
    def is_tracking(self):
        return len(self.live_trackers) > 0

    def create_trackers_from_keypoints(self, tracker_type, key_points, frame):
        for kp in key_points:
            bbox = utils.kp_to_bbox(kp)
            # print(bbox)

            # Initialize tracker with first frame and bounding box
            if not utils.is_bbox_being_tracked(self.live_trackers, bbox):
                self.create_and_add_tracker(tracker_type, frame, bbox)

    def create_and_add_tracker(self, tracker_type, frame, bbox):
        if not bbox:
            raise Exception("null bbox")

        self.total_trackers_started += 1

        tracker = Tracker(self.total_trackers_started, tracker_type, frame, bbox, self.font_size, self.font_colour)
        tracker.update(frame, self.detection_sensitivity)
        self.live_trackers.append(tracker)

    def update_trackers(self, tracker_type, key_points, frame):

        # cache kp -> bbox mapping for removing failed trackers
        kp_bbox_map = {}
        for kp in key_points:
            bbox = utils.kp_to_bbox(kp)
            kp_bbox_map[kp] = bbox

        failed_trackers = []
        for idx, tracker in enumerate(self.live_trackers):

            # Update tracker
            ok, bbox = tracker.update(frame, self.detection_sensitivity)
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
        for kp, new_bbox in kp_bbox_map.items():
            # Hit max trackers?
            if len(self.live_trackers) < self.max_active_trackers:
                if not utils.is_bbox_being_tracked(self.live_trackers, new_bbox):
                    self.create_and_add_tracker(tracker_type, frame, new_bbox)

    def initialise(self, frame, blur, normalise_video):

        self.blur = blur
        self.normalise_video = normalise_video

        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'DASIAMRPN']
        background_subtractor_types = ['KNN']

        self.tracker_type = tracker_types[7]
        self.background_subtractor_type = background_subtractor_types[0]

        source_width = int(frame.shape[1])
        source_height = int(frame.shape[1])

        print(f'Video size w:{source_width} x h:{source_height}')
        if self.normalise_video:
            print(f'Video frames wil be normalised to w: {self.normalised_w_h[0]} x h:{self.normalised_w_h[1]}')

        self.font_size = int(source_height / 1000.0)
        if self.normalise_video:
            self.font_size = int(self.normalised_w_h[1] / 1000.0)

        # MG: Ultimately I would like the visualiser to be passed in
        if self.visualiser is not None:
            self.visualiser.initialise(self.font_size, self.font_colour)

        if self.normalise_video:
            frame = utils.normalize_frame(frame, self.normalised_w_h[0], self.normalised_w_h[1])

        frame_gray = utils.convert_to_gray(frame)

        # Blur image
        if self.blur:
            frame_gray = cv2.GaussianBlur(frame_gray, (self.blur_radius, self.blur_radius), 0)
            # frame_gray = cv2.medianBlur(frame_gray, self.blur_radius)

        self.background_subtractor = BackgroundSubtractorFactory.create(self.background_subtractor_type, self.detection_sensitivity)

        self.frame_output, self.frame_masked_background = utils.apply_background_subtraction(frame_gray, self.background_subtractor, self.mask_pct)

        if self.events is not None:
            self.events.publish_initialise(self.detection_sensitivity, self.blur, self.normalise_video, self.tracker_type, self.background_subtractor_type, source_width, source_height)

    def initialise_background_subtraction(self, frame):

        if self.normalise_video:
            frame = utils.normalize_frame(frame, self.normalised_w_h[0], self.normalised_w_h[1])

        frame_gray = utils.convert_to_gray(frame)

        # Blur image
        if self.blur:
            frame_gray = cv2.GaussianBlur(frame_gray, (self.blur_radius, self.blur_radius), 0)
            # frame_gray = cv2.medianBlur(frame_gray, self.blur_radius)

        self.frame_output, self.frame_masked_background = utils.apply_background_subtraction(frame_gray, self.background_subtractor, self.mask_pct)

    def initialise_trackers(self):

        key_points = utils.perform_blob_detection(self.frame_masked_background, self.detection_sensitivity)

        # Create Trackers
        self.create_trackers_from_keypoints(self.tracker_type, key_points, self.frame_output)

    def process_frame(self, frame, frame_count, fps):

        if self.normalise_video:
            frame = utils.normalize_frame(frame, self.normalised_w_h[0], self.normalised_w_h[1])

        # Copy the frame as we want to mark the original and use the copy for displaying tracking artifacts
        self.frame_output = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur image
        if self.blur:
            frame_gray = cv2.GaussianBlur(frame_gray, (self.blur_radius, self.blur_radius), 0)
            # frame_gray = cv2.medianBlur(frame_gray, self.blur_radius)

        # MG: This needs to be done on an 8 bit gray scale image, the colour image is causing a detection cluster
        _, frame_masked_background = utils.apply_background_subtraction(frame_gray, self.background_subtractor, self.mask_pct)

        # Detect new objects of interest to pass to tracker
        key_points = utils.perform_blob_detection(frame_masked_background, self.detection_sensitivity)

        self.update_trackers(self.tracker_type, key_points, self.frame_output)

        if self.events is not None:
            self.events.publish_process_frame(frame, frame_gray, frame_masked_background, frame_count + 1, self.live_trackers, fps)

        if self.visualiser is not None:
            self.frame_output = self.visualiser.visualise_frame(self, frame, frame_masked_background, self.frame_output, key_points, fps)

        return self.frame_output

    def finalise(self):
        if self.events is not None:
            self.events.publish_finalise(self.total_trackers_started, self.total_trackers_finished)
