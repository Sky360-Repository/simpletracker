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

import time
import math
import uap_tracker.utils as utils
from uap_tracker.tracker_factory import TrackerFactory

class Tracker():

    PROVISIONARY_TARGET = 1
    ACTIVE_TARGET = 2
    LOST_TARGET = 3

    def __init__(self, settings, id, tracker_type, frame, bbox):

        self.settings = settings
        self.id = id
        self.cv2_tracker = TrackerFactory.create(tracker_type)
        self.cv2_tracker.init(frame, bbox)
        self.bboxes = [bbox]
        self.stationary_track_counter = 0
        self.active_track_counter = 0
        self.tracking_state = Tracker.PROVISIONARY_TARGET
        self.bbox_to_check = bbox

        self.start = time.time()
        self.second_counter = 0
        self.tracked_boxes = [bbox]

    # (x1,y1,w,h)
    def get_bbox(self):
        return self.bboxes[-1]

    def get_center(self):
        x1, y1, w, h = self.get_bbox()
        return (int(x1+(w/2)),
                int(y1+(h/2)))

    def update(self, frame, validate_target=True):
        ok, bbox = self.cv2_tracker.update(frame)
        # print(f'updating tracker {self.id}, result: {ok}')
        if ok:
            self.bboxes.append(bbox)

            if validate_target:

                validate_bbox = False
                if math.floor((time.time() - self.start)) > self.second_counter:
                    self.tracked_boxes.append(bbox)
                    self.second_counter = self.second_counter + 1
                    validate_bbox = True

                # print(f'Elaspsed seconds: {math.floor((iteration - self.start))}s - iterate: {iterate}')

                stationary_check_threshold = self.settings['stationary_check_threshold']
                stationary_scavanage_threshold = math.floor(stationary_check_threshold * 1.25)
                orphaned_check_thold = self.settings['orphaned_check_threshold']

                if len(self.tracked_boxes) > 1:
                    # MG: if the item being tracked has moved out of its initial bounds, then it's a trackable target
                    if utils.bbox_overlap(self.bbox_to_check, bbox) == 0.0:
                        # self.bbox_color = self.font_color
                        if self.tracking_state != Tracker.ACTIVE_TARGET:
                            self.tracking_state = Tracker.ACTIVE_TARGET
                            self.bbox_to_check = bbox

                    if validate_bbox:
                        # print(f'5 X --> tracker {self.id}, total length: {len(self.bboxes)}')
                        previous_tracked_bbox = self.tracked_boxes[-1]
                        if utils.bbox_overlap(self.bbox_to_check, previous_tracked_bbox) > 0:
                            # MG: this bounding box has remained pretty static, its now closer to getting scavenged
                            self.stationary_track_counter += 1
                        else:
                            self.stationary_track_counter = 0

                if stationary_check_threshold <= self.stationary_track_counter < stationary_scavanage_threshold:
                    # MG: If its not moved enough then mark it as red for potential scavenging
                    self.tracking_state = Tracker.LOST_TARGET
                    # print(f'>> updating tracker {self.id} state to LOST_TARGET')
                elif self.stationary_track_counter >= stationary_scavanage_threshold:
                    print(f'Scavenging tracker {self.id}')
                    ok = False

                if self.tracking_state == Tracker.ACTIVE_TARGET:
                    self.active_track_counter += 1
                    if self.active_track_counter > orphaned_check_thold:
                        self.bbox_to_check = bbox
                        self.active_track_counter = 0

        return ok, bbox

    def is_tracking(self):
        return self.tracking_state == Tracker.ACTIVE_TARGET

    def does_bbx_overlap(self, bbox):
        overlap = utils.bbox_overlap(self.bboxes[-1], bbox)
        # print(f'checking tracking overlap {overlap} for {self.id}')
        return overlap > 0

    def is_bbx_contained(self, bbox):
        return utils.bbox1_contain_bbox2(self.bboxes[-1], bbox)

    def bbox_color(self):
        return {
            Tracker.PROVISIONARY_TARGET: (25, 175, 175),
            Tracker.ACTIVE_TARGET: (50, 170, 50),
            Tracker.LOST_TARGET: (50, 50, 225)
        }[self.tracking_state]
