import cv2
import numpy as np
import uap_tracker.utils as utils
from uap_tracker.tracker_factory import TrackerFactory

#
# Tracks a single object
#


class Tracker():

    PROVISIONARY_TARGET = 1
    ACTIVE_TARGET = 2
    LOST_TARGET = 3

    def __init__(self, id, tracker_type, frame, bbox):

        self.id = id
        self.cv2_tracker = TrackerFactory.create(tracker_type)
        self.cv2_tracker.init(frame, bbox)
        self.bbox_start = bbox
        # self.bbox_slide = bbox
        self.bboxes = [bbox]
        self.frame_stationary_check = 0
        self.tracking_state = Tracker.PROVISIONARY_TARGET

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
                stationary_check_thold = 4
                stationary_check_max = 5

                # MG: Default to yellow as it needs to prove itself as a target first
                if len(self.bboxes) <= 10:
                    # self.bbox_color = (25, 175, 175)
                    self.tracking_state = Tracker.PROVISIONARY_TARGET
                    # print(f'>> updating tracker {self.id} state to PROVISIONARY_TARGET')
                elif len(self.bboxes) > 10:

                    # MG: if the item being tracked has moved out of its initial bounds, then it's a trackable target
                    if utils.bbox_overlap(self.bbox_start, bbox) == 0.0:
                        # self.bbox_color = self.font_color
                        if self.tracking_state != Tracker.ACTIVE_TARGET:
                            self.tracking_state = Tracker.ACTIVE_TARGET
                            # print(f'>> updating tracker {self.id} state to ACTIVE_TARGET')
                            self.bbox_start = bbox

                    mod = float(len(self.bboxes) % 5)
                    if mod == 0:
                        # print(f'5 X --> tracker {self.id}, total length: {len(self.bboxes)}')
                        bbox_2 = self.bboxes[-5]
                        if utils.bbox_overlap(self.bbox_start, bbox_2) > 0:
                            # MG: this bounding box has remained pretty static, its now closer to getting scavenged
                            self.frame_stationary_check += 1
                        else:
                            self.frame_stationary_check = 0

                if stationary_check_thold <= self.frame_stationary_check < stationary_check_max:
                    # MG: If its not moved enough then mark it as red for potential scavenging
                    # self.bbox_color = (50, 50, 225)
                    self.tracking_state = Tracker.LOST_TARGET
                    # print(f'>> updating tracker {self.id} state to LOST_TARGET')
                elif self.frame_stationary_check >= stationary_check_max:
                    print(f'Scavenging tracker {self.id}')
                    ok = False

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
        if self.tracking_state == Tracker.PROVISIONARY_TARGET:
            return (25, 175, 175)
        elif self.tracking_state == Tracker.LOST_TARGET:
            return (50, 50, 225)
        else:
            return (50, 170, 50)
