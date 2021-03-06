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
import numpy as np
import uap_tracker.utils as utils

class Mask():

    def __init__(self):
        pass

    def initialise(self, init_frame):
        pass

    def apply(self, frame):
        pass

class FisheyeMask(Mask):

    def __init__(self, mask_pct):
        self.mask_pct = mask_pct

    def initialise(self, init_frame):
        self.mask_height = (100 - self.mask_pct) / 100.0
        self.mask_radius = self.mask_height / 2.0
        self.shape = init_frame.shape[:2]
        self.height = self.shape[0]
        self.width = self.shape[1]
        self.new_width = int(self.mask_height * self.width)
        self.new_height = int(self.mask_height * self.height)
        return (self.new_width, self.new_height)

    def apply(self, frame):
        mask = np.zeros(self.shape, dtype=np.uint8)
        cv2.circle(mask, (int(self.width / 2), int(self.height / 2)), int(min(self.height, self.width) * self.mask_radius), 255, -1)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        clipped_masked_frame = utils.clip_at_center(
            masked_frame,
            (int(self.width / 2), int(self.height / 2)),
            self.width,
            self.height,
            self.new_width,
            self.new_height)
        return clipped_masked_frame