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

import cv2
import time
import math
import uap_tracker.utils as utils
from uap_tracker.tracker_factory import TrackerFactory
from uap_tracker.track_prediction import TrackPrediction

########################################################################################################################
# This class represents a single target/blob that has been identified on the frame and is currently being tracked      #
# If track validation is turned on then there are a couple of further steps that are taken in order to ensure the      #
# target is valid and not a phantom target. This logic is by no means perfect or bullet proof but as tracking an       #
# object is the most expensive in terms of processing time, we need only try and track targets that have the potential # 
# to be good and valid targets.                                                                                        #
########################################################################################################################
class Tracker():

    PROVISIONARY_TARGET = 1
    ACTIVE_TARGET = 2
    LOST_TARGET = 3

    def __init__(self, settings, id, frame, bbox):

        self.settings = settings
        self.id = id
        self.cv2_tracker = TrackerFactory.create(settings)
        # print(f"--> bbox:{bbox}")
        self.cv2_tracker.init(frame, bbox)
        self.bboxes = [bbox]
        self.stationary_track_counter = 0
        self.active_track_counter = 0
        self.tracking_state = Tracker.PROVISIONARY_TARGET
        self.bbox_to_check = bbox
        
        self.start = time.time()
        self.second_counter = 0
        self.tracked_boxes = [bbox]
        self.center_points = []        

        self.track_predictor = TrackPrediction(id, bbox)
        self.predictor_center_points = []

    # function to get the latest bbox in the format (x1,y1,w,h)
    def get_bbox(self):
        return self.bboxes[-1]

    # function to determine the center of the the bbox being tracked
    def get_center(self):
        x1, y1, w, h = self.get_bbox()
        return (int(x1+(w/2)),
                int(y1+(h/2)))

    # function to update the bbox on the frame, also if validation is enabled then some addtional logic is executed to determine 
    # if the target is still a avlid target
    def update(self, frame):
        ok, bbox = self.cv2_tracker.update(frame)
        # print(f'updating tracker {self.id}, result: {ok}')
        if ok:
            self.bboxes.append(bbox)

            # Mike: If we have track plotting enabled, then we need to store the center points of the bboxes so that we can plo the 
            # entire track on the frame including the colour
            if self.settings['track_plotting_enabled']:
                self.center_points.append((self.get_center(), self.bbox_color()))

            if self.settings['track_prediction_enabled']:
                self.predictor_center_points.append(self.track_predictor.update(bbox))

            if self.settings['enable_track_validation']:

                # Mike: perform validation logic every second, on the tickover of that second. Validation logic is very much dependent on the 
                # target moving a certain amount over time. The technology that we use does have its limitation in that it will 
                # identify and try and track false positives. This validaiton logic is in place to try and limit this
                validate_bbox = False
                if math.floor((time.time() - self.start)) > self.second_counter:
                    self.tracked_boxes.append(bbox)
                    self.second_counter = self.second_counter + 1
                    validate_bbox = True

                # print(f'Elaspsed seconds: {math.floor((iteration - self.start))}s - iterate: {iterate}')

                # Mike: grab the validation config options from the settings dictionary
                stationary_track_threshold = self.settings['stationary_track_threshold']
                stationary_scavanage_threshold = math.floor(stationary_track_threshold * 1.5)
                orphaned_track_thold = self.settings['orphaned_track_threshold']

                # Mike: Only process validation after a second, we need to allow the target to move
                if len(self.tracked_boxes) > 1:
                    # Mike: if the item being tracked has moved out of its initial bounds, then it's an active target
                    if utils.bbox_overlap(self.bbox_to_check, bbox) == 0.0:
                        if self.tracking_state != Tracker.ACTIVE_TARGET:
                            self.tracking_state = Tracker.ACTIVE_TARGET
                            self.bbox_to_check = bbox
                            self.stationary_track_counter = 0

                    if validate_bbox:
                        # print(f'5 X --> tracker {self.id}, total length: {len(self.bboxes)}')
                        previous_tracked_bbox = self.tracked_boxes[-1]
                        if utils.bbox_overlap(self.bbox_to_check, previous_tracked_bbox) > 0:
                            # Mike: this bounding box has remained pretty static, its now closer to getting scavenged
                            self.stationary_track_counter += 1
                        else:
                            self.stationary_track_counter = 0

                # Mike: If the target has not moved for a period of time, we classify the target as lost
                if stationary_track_threshold <= self.stationary_track_counter < stationary_scavanage_threshold:
                    # Mike: If its not moved enough then mark it as red for potential scavenging
                    self.tracking_state = Tracker.LOST_TARGET
                    # print(f'>> updating tracker {self.id} state to LOST_TARGET')
                elif self.stationary_track_counter >= stationary_scavanage_threshold:
                    # print(f'Scavenging tracker {self.id}')
                    # Mike: If it has remained stationary for a period of time then we are no longer interested
                    ok = False

                #Mike: If its an active target then update counters at the popint of validation
                if self.tracking_state == Tracker.ACTIVE_TARGET:
                    self.active_track_counter += 1
                    if self.active_track_counter > orphaned_track_thold:
                        self.bbox_to_check = bbox
                        self.active_track_counter = 0

        return ok, bbox

    # Utility function to determine is this tracker has an active target
    def is_tracking(self):
        return self.tracking_state == Tracker.ACTIVE_TARGET

    # Utility function to determine if there is overlap between existing and new bboxes
    def does_bbx_overlap(self, bbox):
        overlap = utils.bbox_overlap(self.bboxes[-1], bbox)
        # print(f'checking tracking overlap {overlap} for {self.id}')
        return overlap > 0
    
    # Utility function to determine if there is containment of new bboxes
    def is_bbx_contained(self, bbox):
        return utils.bbox1_contain_bbox2(self.bboxes[-1], bbox)

    # Utility function to provide the colour of the bbox on the frame
    def bbox_color(self):
        return {
            Tracker.PROVISIONARY_TARGET: (25, 175, 175),
            Tracker.ACTIVE_TARGET: (50, 170, 50),
            Tracker.LOST_TARGET: (50, 50, 225)
        }[self.tracking_state]
