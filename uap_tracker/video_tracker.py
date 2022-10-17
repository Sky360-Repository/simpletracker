# Original work Copyright (c) 2022 Sky360
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

import numpy as np
from threading import Thread
from uap_tracker.stopwatch import Stopwatch
import uap_tracker.utils as utils
from uap_tracker.tracker import Tracker

################################################################################################
# This class is pretty much considered the application part of the Simple Tracker application. #
################################################################################################
class VideoTracker():

    DETECTION_SENSITIVITY_HIGH = 1
    DETECTION_SENSITIVITY_NORMAL = 2
    DETECTION_SENSITIVITY_LOW = 3

    FRAME_TYPE_ANNOTATED = 'annotated'
    FRAME_TYPE_GREY = 'grey'
    FRAME_TYPE_MASKED_BACKGROUND = 'masked_background'
    FRAME_TYPE_OPTICAL_FLOW = 'optical_flow'
    FRAME_TYPE_ORIGINAL = 'original'

    def __init__(self, settings, events, visualizer):
      
        self.settings = settings
        self.total_trackers_finished = 0
        self.total_trackers_started = 0
        self.live_trackers = []
        self.events = events
        self.visualizer = visualizer
        self.frame_output = None
        self.frame_masked_background = None
        self.frames = {}
        self.keypoints = []

        print(
            f"Initializing Tracker:\n  resize_frame:{self.settings['resize_frame']}\n  resize_dimension:{self.settings['resize_dimension']}\n  noise_reduction: {self.settings['noise_reduction']}\n  mask_type:{self.settings['mask_type']}\n  mask_pct:{self.settings['mask_pct']}\n  sensitivity:{self.settings['detection_sensitivity']}\n  max_active_trackers:{self.settings['max_active_trackers']}\n  tracker_type:{self.settings['tracker_type']}")

    @property
    def is_tracking(self):
        return len(self.live_trackers) > 0

    def active_trackers(self):
        trackers = filter(lambda x: x.is_tracking(), self.live_trackers)
        if trackers is None:
            return []
        else:
            return trackers

    # function to create trackers from extracted keypoints
    def create_trackers_from_keypoints(self, tracker_type, key_points, frame):
        for kp in key_points:
            bbox = utils.kp_to_bbox(kp)
            # print(bbox)

            # Initialize tracker with first frame and bounding box
            if not utils.is_bbox_being_tracked(self.live_trackers, bbox):
                self.create_and_add_tracker(tracker_type, frame, bbox)

    # function to create an add the tracker to the list of active trackers
    def create_and_add_tracker(self, frame, bbox):
        if not bbox:
            raise Exception("null bbox")

        self.total_trackers_started += 1

        tracker = Tracker(self.settings, self.total_trackers_started, frame, bbox)
        tracker.update(frame)
        self.live_trackers.append(tracker)

    # function to update existing trackers and and it a target is not tracked then create a new tracker to track the target
    #
    # trackers are run on seperate threads in the hope to speed up the applicaiton by taking advantage of parallelism
    def update_trackers(self, bboxes, frame):

        unmatched_bboxes = bboxes.copy()
        failed_trackers = []
        tracker_count = len(self.live_trackers)

        threads = [None] * tracker_count
        results = [None] * tracker_count

        # Mike: We can do the tracker updates in parallel
        # This will not work due to the GIL, need to split this processing using a different approach
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
            if len(self.live_trackers) < self.settings['max_active_trackers']:
                if not utils.is_bbox_being_tracked(self.live_trackers, new_bbox):
                    self.create_and_add_tracker(frame, new_bbox)

    # function to initialise objects, using cuda streams apparently its important to allocated memory once versus over and over 
    # again as it improves performance
    def initialise(self, frame_proc, init_frame):

        # Mike: Initiliase the processor as well
        size = frame_proc.initialise(init_frame)
        size = (size[1], size[0], 3)

        #print(f'pre allocate size {size}')

        # Mike: Preallocate the frames dictionary
        self.frames = {
            self.FRAME_TYPE_GREY: np.empty(size[0:2], np.uint8),
            self.FRAME_TYPE_MASKED_BACKGROUND: np.empty(size[0:2], np.uint8),
            self.FRAME_TYPE_OPTICAL_FLOW: np.empty(size, np.uint8),
            self.FRAME_TYPE_ORIGINAL: np.empty(size, np.uint8)
        }

    # main entry point of the application although it generally gets handed over to an specific implimentation of the
    # frame processor as soon as
    def process_frame(self, frame_proc, frame, frame_count, fps, stream=None):

        self.fps = fps
        self.frame_count = frame_count
        self.frames[self.FRAME_TYPE_ANNOTATED] = None

        with Stopwatch(mask='Frame '+str(frame_count)+': Took {s:0.4f} seconds to process', enable=self.settings['enable_stopwatch']):

            self.keypoints = frame_proc.process_frame(self, frame, frame_count, fps, stream)

            if self.events is not None:
                self.events.publish_process_frame(self)

            if self.visualizer is not None:
                self.visualizer.Visualize(self)

    # essentially shutdown the tracking applicaiton and clean up after yourself
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
        annotated_frame = self.get_image(self.FRAME_TYPE_ANNOTATED)
        if annotated_frame is None:
            annotated_frame = self.frames[self.FRAME_TYPE_ORIGINAL].copy()
            if active_trackers_only:
                for tracker in self.active_trackers():
                    utils.add_bbox_to_image(tracker.get_bbox(), annotated_frame, tracker.id, 1, tracker.bbox_color(), self.settings)
                    if self.settings['track_plotting_enabled']:
                        utils.add_track_line_to_image(tracker, annotated_frame)
                    if self.settings['track_prediction_enabled']:
                        utils.add_predicted_point_to_image(tracker, annotated_frame)                        
            else:
                for tracker in self.live_trackers:
                    utils.add_bbox_to_image(tracker.get_bbox(), annotated_frame, tracker.id, 1, tracker.bbox_color(), self.settings)
                    if self.settings['track_plotting_enabled']:
                        utils.add_track_line_to_image(tracker, annotated_frame)
                    if self.settings['track_prediction_enabled']:
                        utils.add_predicted_point_to_image(tracker, annotated_frame)

            self.frames[self.FRAME_TYPE_ANNOTATED] = annotated_frame

        return self.frames[self.FRAME_TYPE_ANNOTATED]

    # utility function to add a frame to the dictionary for usage by listeners / visualizers
    def add_image(self, frame_name, frame):
        self.frames[frame_name] = frame

    # funtions mainly called from listeners / visualizers

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
