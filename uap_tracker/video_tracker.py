import cv2
import numpy as np
from threading import Thread
from uap_tracker.stopwatch import Stopwatch
#import multiprocessing
import uap_tracker.utils as utils
from uap_tracker.tracker import Tracker

#
# Tracks multiple objects in a video
#


class VideoTracker():

    DETECTION_SENSITIVITY_HIGH = 1
    DETECTION_SENSITIVITY_NORMAL = 2
    DETECTION_SENSITIVITY_LOW = 3

    FRAME_TYPE_ANNOTATED = 'annotated'
    FRAME_TYPE_GREY = 'grey'
    FRAME_TYPE_MASKED_BACKGROUND = 'masked_background'
    FRAME_TYPE_OPTICAL_FLOW = 'optical_flow'
    FRAME_TYPE_ORIGINAL = 'original'

    def __init__(self, detection_mode, events, visualizer, detection_sensitivity=2, mask_pct=8, noise_reduction=True, resize_frame=True,
                 resize_dim=1024, calculate_optical_flow=True, max_active_trackers=10, tracker_type='CSRT'):

        print(
            f"Initializing Tracker:\n  resize_frame:{resize_frame}\n  resize_dim:{resize_dim}\n  noise_reduction: {noise_reduction}\n  mask_pct:{mask_pct}\n  sensitivity:{detection_sensitivity}\n  max_active_trackers:{max_active_trackers}\n  tracker_type:{tracker_type}")

        self.detection_mode = detection_mode
        if detection_sensitivity < 1 or detection_sensitivity > 3:
            raise Exception(
                f"Unknown sensitivity option ({detection_sensitivity}). 1, 2 and 3 is supported not {detection_sensitivity}.")

        self.detection_sensitivity = detection_sensitivity
        self.total_trackers_finished = 0
        self.total_trackers_started = 0
        self.live_trackers = []
        self.events = events
        self.visualizer = visualizer
        self.max_active_trackers = max_active_trackers
        self.mask_pct = mask_pct
        self.calculate_optical_flow = calculate_optical_flow
        self.noise_reduction = noise_reduction
        self.resize_frame = resize_frame
        self.frame_output = None
        self.frame_masked_background = None
        self.tracker_type = tracker_type
        self.resize_dim = resize_dim
        self.frames = {}
        self.keypoints = []

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

        unmatched_bboxes = bboxes.copy()
        failed_trackers = []
        tracker_count = len(self.live_trackers)

        threads = [None] * tracker_count
        results = [None] * tracker_count

        # Mike: We can do the tracker updates in parallel
        for i in range(tracker_count):
            tracker = self.live_trackers[i]
            threads[i] = Thread(target=self.update_tracker_task, args=(tracker, frame, results, i))
            threads[i].start()

        # Mike: We got to wait for the threads to join before we proceed
        for i in range(tracker_count):
            threads[i].join()

        for i in range(tracker_count):
            tracker = self.live_trackers[i]
            ok, bbox = results[i]
            if not ok:
                # Tracking failure
                failed_trackers.append(tracker)

            # Try to match the new detections with this tracker
            for new_bbox in bboxes:
                if new_bbox in unmatched_bboxes:
                    overlap = utils.bbox_overlap(bbox, new_bbox)
                    # print(f'Overlap: {overlap}; bbox:{bbox}, new_bbox:{new_bbox}')
                    if overlap > 0.2:
                        unmatched_bboxes.remove(new_bbox)

        # remove failed trackers from live tracking
        for tracker in failed_trackers:
            self.live_trackers.remove(tracker)
            self.total_trackers_finished += 1

        # Add new detections to live tracker
        for new_bbox in unmatched_bboxes:
            # Hit max trackers?
            if len(self.live_trackers) < self.max_active_trackers:
                if not utils.is_bbox_being_tracked(self.live_trackers, new_bbox):
                    self.create_and_add_tracker(tracker_type, frame, new_bbox)

    def process_frame(self, frame_proc, frame, frame_grey, frame_masked_background, keypoints, frame_count, fps, stream=None):

        self.fps = fps
        self.frame_count = frame_count
        self.frames.clear()

        with Stopwatch(mask='Frame '+str(frame_count)+': Took {s:0.4f} seconds to process', quiet=True):

            self.keypoints = frame_proc.process_frame(self, frame, frame_grey, frame_masked_background, keypoints, frame_count, fps, stream)

            if self.events is not None:
                self.events.publish_process_frame(self)

            if self.visualizer is not None:
                self.visualizer.Visualize(self)

    def finalise(self):
        if self.events is not None:
            self.events.publish_finalise(
                self.total_trackers_started, self.total_trackers_finished)

    # called from listeners / visualizers
    # returns named image for current frame
    def get_image(self, frame_name):
        frame = None
        if frame_name in self.frames:
            frame = self.frames[frame_name]
        return frame

    # called from listeners / visualizers
    # returns annotated image for current frame
    def get_annotated_image(self, active_trackers_only=True):
        annotated_frame = self.frames.get(self.FRAME_TYPE_ANNOTATED, None)
        if annotated_frame is None:
            annotated_frame = self.frames[self.FRAME_TYPE_ORIGINAL].copy()
            if active_trackers_only:
                for tracker in self.active_trackers():
                    utils.add_bbox_to_image(tracker.get_bbox(), annotated_frame, tracker.id, 1, tracker.bbox_color())
            else:
                for tracker in self.live_trackers:
                    utils.add_bbox_to_image(tracker.get_bbox(), annotated_frame, tracker.id, 1, tracker.bbox_color())
            self.frames[self.FRAME_TYPE_ANNOTATED] = annotated_frame

        return self.frames[self.FRAME_TYPE_ANNOTATED]

    def add_image(self, frame_name, frame):
        self.frames[frame_name] = frame

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

    # Mike: Identifying tasks that can be called on seperate threads to try and speed this sucker up
    def update_tracker_task(self, tracker, frame, results, index):
        results[index] = tracker.update(frame)
