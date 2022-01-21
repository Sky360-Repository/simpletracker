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

import uap_tracker.utils as utils
from uap_tracker.visualizer import Visualizer


class TwoByTwoOpticalFlowVisualiser(Visualizer):

    def visualise_frame(self, video_tracker):

        frame_input = video_tracker.get_image(video_tracker.FRAME_TYPE_ORIGINAL)
        optical_flow_frame = video_tracker.get_image(video_tracker.FRAME_TYPE_OPTICAL_FLOW)
        frame_output = video_tracker.get_annotated_image()
        fps = video_tracker.get_fps()

        utils.stamp_original_frame(
            frame_input, self.font_size, self.font_colour)

        utils.stamp_output_frame(
            video_tracker, frame_output, self.font_size, self.font_colour, fps)

        bottom_left_frame = optical_flow_frame
        bottom_right_frame = optical_flow_frame

        return utils.combine_frames_2x2(
            frame_input, frame_output, bottom_left_frame, bottom_right_frame
        )
