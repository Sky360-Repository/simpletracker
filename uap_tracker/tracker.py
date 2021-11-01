import cv2
import numpy as np
import uap_tracker.utils as utils
from uap_tracker.tracker_factory import TrackerFactory

#
# Tracks a single object
#
class Tracker():

    def __init__(self, id, tracker_type, frame, frame_hsv, bbox, font_size, font_color):

        self.id = id
        self.cv2_tracker = TrackerFactory.create(tracker_type)
        self.cv2_tracker.init(frame, bbox)
        self.bboxes = [bbox]
        self.font_size = font_size
        self.font_color = font_color

    def get_bbox(self):
        return self.bboxes[-1]

    def update(self, frame, frame_hsv, sensitivity):
        ok, bbox = self.cv2_tracker.update(frame)
        if ok:
            self.bboxes.append(bbox)
            utils.add_bbox_to_image(bbox, frame, self.id, self.font_size, self.font_color)

        return ok, bbox

    def does_bbx_overlap(self, bbox):
        overlap = utils.bbox_overlap(self.bboxes[-1], bbox)
        # print(f'checking tracking overlap {overlap} for {self.id}')
        return overlap > 0

    def is_bbx_contained(self, bbox):
        return utils.bbox1_contain_bbox2(self.bboxes[-1], bbox)
