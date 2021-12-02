import cv2
import numpy as np
import time
import uap_tracker.utils as utils
from uap_tracker.tracker import Tracker
from uap_tracker.background_subtractor_factory import BackgroundSubtractorFactory
from uap_tracker.dense_optical_flow import DenseOpticalFlow

#
# Tracks multiple objects in a video
#


class VideoTracker():

    DETECTION_SENSITIVITY_HIGH = 1
    DETECTION_SENSITIVITY_NORMAL = 2
    DETECTION_SENSITIVITY_LOW = 3

    def __init__(self, detection_mode, events, detection_sensitivity=2, mask_pct=8, noise_reduction=True, normalise_video=True, calculate_optical_flow=True):

        print(
            f"Initializing Tracker:\n  normalize:{normalise_video}\n  noise_reduction: {noise_reduction}\n  mask_pct:{mask_pct}\n  sensitivity:{detection_sensitivity}")

        self.detection_mode = detection_mode
        if detection_sensitivity < 1 or detection_sensitivity > 3:
            raise Exception(
                f"Unknown sensitivity option ({detection_sensitivity}). 1, 2 and 3 is supported not {detection_sensitivity}.")

        self.detection_sensitivity = detection_sensitivity
        self.total_trackers_finished = 0
        self.total_trackers_started = 0
        self.live_trackers = []
        self.events = events
        self.normalised_w_h = (1024, 1024)
        self.blur_radius = 3
        self.max_active_trackers = 10
        self.mask_pct = mask_pct
        self.calculate_optical_flow = calculate_optical_flow

        self.noise_reduction = noise_reduction
        self.normalise_video = normalise_video
        self.tracker_type = None
        self.background_subtractor_type = None
        self.background_subtractor = None
        self.frame_output = None
        self.frame_masked_background = None

        self.dof = DenseOpticalFlow(480, 480)

        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                         'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'DASIAMRPN']
        background_subtractor_types = ['KNN']

        self.tracker_type = tracker_types[7]
        self.background_subtractor_type = background_subtractor_types[0]

        self.background_subtractor = BackgroundSubtractorFactory.create(
            self.background_subtractor_type, self.detection_sensitivity)

    @property
    def is_tracking(self):
        return len(self.live_trackers) > 0

    def active_trackers(self):
        trackers = filter(lambda x: x.is_tracking(), self.live_trackers)
        if trackers is None:
            return []
        else:
            return trackers

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

        tracker = Tracker(self.total_trackers_started, tracker_type,
                          frame, bbox)
        tracker.update(frame)
        self.live_trackers.append(tracker)

    def update_trackers(self, tracker_type, bboxes, frame):
        tic1 = time.perf_counter()
        unmatched_bboxes = bboxes.copy()
        tic2 = time.perf_counter()
        print(f"  Copying bboxes {tic2 - tic1:0.4f} seconds")

        failed_trackers = []
        for tracker in self.live_trackers:

            # Update tracker
            tic3 = time.perf_counter()
            ok, bbox = tracker.update(frame)
            tic4 = time.perf_counter()
            print(f"  Updating tracker {tracker.id} took {tic4 - tic3:0.4f} seconds")

            if not ok:
                # Tracking failure
                failed_trackers.append(tracker)

            # Try to match the new detections with this tracker
            #if ok:
            tic5 = time.perf_counter()
            for new_bbox in bboxes:
                if new_bbox in unmatched_bboxes:
                    overlap = utils.bbox_overlap(bbox, new_bbox)
                    # print(f'Overlap: {overlap}; bbox:{bbox}, new_bbox:{new_bbox}')
                    if overlap > 0.2:
                        unmatched_bboxes.remove(new_bbox)
            tic6 = time.perf_counter()
            print(f"  Matching new detections took {tic6 - tic5:0.4f} seconds")

        # remove failed trackers from live tracking
        for tracker in failed_trackers:
            self.live_trackers.remove(tracker)
            self.total_trackers_finished += 1

        # Add new detections to live tracker
        tic7 = time.perf_counter()
        for new_bbox in unmatched_bboxes:
            # Hit max trackers?
            if len(self.live_trackers) < self.max_active_trackers:
                if not utils.is_bbox_being_tracked(self.live_trackers, new_bbox):
                    self.create_and_add_tracker(tracker_type, frame, new_bbox)
        tic8 = time.perf_counter()
        print(f"  Check max trackers and overlap took {tic8 - tic7:0.4f} seconds")

    def process_frame(self, frame, frame_count, fps):
        # print(f" fps:{int(fps)}", end='\r')
        self.fps = fps
        self.frame_count = frame_count

        tic1 = time.perf_counter()

        frame = utils.apply_fisheye_mask(frame, self.mask_pct)
        tic2 = time.perf_counter()
        print(f"{frame_count}: Applying fisheye mask {tic2 - tic1:0.4f} seconds")

        frame = utils.resize_frame(self.normalise_video, frame, self.normalised_w_h[0], self.normalised_w_h[1])
        tic3 = time.perf_counter()
        print(f"{frame_count}: Resizing frame {tic3 - tic2:0.4f} seconds")

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tic4 = time.perf_counter()
        print(f"{frame_count}: Converting to gray scale {tic4 - tic3:0.4f} seconds")

        frame_gray = utils.noise_reduction(self.noise_reduction, frame_gray, self.blur_radius)
        tic5 = time.perf_counter()
        print(f"{frame_count}: Reducing noise {tic5 - tic4:0.4f} seconds")

        self.frames = {
            'original': frame,
            'grey': frame_gray
        }
        if self.detection_mode == 'background_subtraction':
            tic6 = time.perf_counter()
            keypoints, frame_masked_background = self.keypoints_from_bg_subtraction(
                frame_gray)
            if frame_count < 5:
                # Need 5 frames to get the background subtractor initialised
                return
            self.frames['masked_background'] = frame_masked_background
            bboxes = [utils.kp_to_bbox(x) for x in keypoints]
            tic7 = time.perf_counter()
            print(f"{frame_count}: Background subtraction {tic7 - tic6:0.4f} seconds")

            if self.calculate_optical_flow:
                self.frames['optical_flow'] = self.optical_flow(frame_gray)
                tic8 = time.perf_counter()
                print(f"{frame_count}: Calculating opticalflow {tic8 - tic7:0.4f} seconds")
        else:
            bboxes = []
            keypoints = []

        self.keypoints = keypoints
        tic9 = time.perf_counter()

        self.update_trackers(self.tracker_type, bboxes, frame)
        tic10 = time.perf_counter()
        print(f"{frame_count}: Updating trackers {tic10 - tic9:0.4f} seconds")

        frame_count + 1

        if self.events is not None:
            self.events.publish_process_frame(self)

        tic11 = time.perf_counter()
        print(f"{frame_count}: Publishing process frame event {tic11 - tic10:0.4f} seconds")

    def optical_flow(self, frame_gray):
        dof_frame = self.dof.process_grey_frame(frame_gray)
        height, width = frame_gray.shape
        return cv2.resize(dof_frame, (width, height))

    def keypoints_from_bg_subtraction(self, frame_gray):
        # MG: This needs to be done on an 8 bit gray scale image, the colour image is causing a detection cluster
        frame_masked_background = utils.apply_background_subtraction(
            frame_gray, self.background_subtractor)

        # Detect new objects of interest to pass to tracker
        key_points = utils.perform_blob_detection(
            frame_masked_background, self.detection_sensitivity)
        return key_points, frame_masked_background

    def finalise(self):
        if self.events is not None:
            self.events.publish_finalise(
                self.total_trackers_started, self.total_trackers_finished)

    # called from listeners / visualizers
    # returns named image for current frame
    def get_image(self, frame_name):
        return self.frames[frame_name]

    # called from listeners / visualizers
    # returns annotated image for current frame
    def get_annotated_image(self, active_trackers_only=True):
        annotated_frame = self.frames.get('annotated_image', None)
        if annotated_frame is None:
            annotated_frame = self.frames['original'].copy()
            if active_trackers_only:
                for tracker in self.active_trackers():
                    utils.add_bbox_to_image(tracker.get_bbox(), annotated_frame, tracker.id, 1, tracker.bbox_color())
            else:
                for tracker in self.live_trackers:
                    utils.add_bbox_to_image(tracker.get_bbox(), annotated_frame, tracker.id, 1, tracker.bbox_color())
            self.frames['annotated_image'] = annotated_frame

        return self.frames['annotated_image']

    # called from listeners / visualizers
    # returns all images for current frame

    def get_images(self):
        return self.frames

    def get_fps(self):
        return int(self.fps)

    def get_frame_count(self):
        return self.frame_count

    def get_live_trackers(self):
        return self.live_trackers

    def get_keypoints(self):
        return self.keypoints
