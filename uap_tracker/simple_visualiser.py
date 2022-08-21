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
from visualizer import Visualizer


class SimpleVisualiser(Visualizer):
        
    def visualise_frame(self, video_tracker):
        frame_output = video_tracker.get_annotated_image(active_trackers_only=False)
        fps = video_tracker.get_fps()
        utils.stamp_output_frame(video_tracker, frame_output, self.font_size, self.font_colour, fps, self.font_thickness)
        return frame_output
