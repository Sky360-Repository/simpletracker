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
from uap_tracker.visualizer import Visualizer


class TwoByTwoVisualiser(Visualizer):

    def visualise_frame(self, video_tracker):
        frame_input = video_tracker.get_image(video_tracker.FRAME_TYPE_ORIGINAL)
        frame_masked_background = video_tracker.get_image(video_tracker.FRAME_TYPE_MASKED_BACKGROUND)
        frame_output = video_tracker.get_annotated_image(active_trackers_only=False)

        key_points = video_tracker.get_keypoints()
        fps = video_tracker.get_fps()

        utils.stamp_original_frame(
            frame_input, self.font_size, self.font_colour)

        utils.stamp_output_frame(
            video_tracker, frame_output, self.font_size, self.font_colour, fps)

        # Create a copy as we need to put text on it and also turn it into a 24 bit image
        frame_masked_background_copy = cv2.cvtColor(frame_masked_background, cv2.COLOR_GRAY2BGR)

        frame_masked_background_with_key_points = cv2.drawKeypoints(
            frame_masked_background_copy, key_points, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        msg = f"Detected {len(key_points)} Key Points (Sky360)"
        cv2.putText(frame_masked_background_with_key_points, msg, (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)

        cv2.putText(frame_masked_background_copy, "Masked Background (Sky360)",
                    (100, 200), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)
        bottom_left_frame = frame_masked_background_copy
        bottom_right_frame = frame_masked_background_with_key_points

        return utils.combine_frames_2x2(frame_input, frame_output, bottom_left_frame, bottom_right_frame)
