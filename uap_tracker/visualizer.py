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

from uap_tracker import utils


class Visualizer():
    def __init__(self, max_display_dim, font_size, font_thickness):
        self.max_display_dim = max_display_dim
        self.font_size = font_size
        self.font_thickness = font_thickness
        self.font_colour = (50, 170, 50)

    def display(self, frame):
        # Display result, resize it to a standard size
        utils.display_frame(frame, self.max_display_dim)

    def Visualize(self, video_tracker):
        msg = f"Trackers: trackable:{sum(map(lambda x: x.is_tracking(), video_tracker.live_trackers))}, alive:{len(video_tracker.live_trackers)}, started:{video_tracker.total_trackers_started}, ended:{video_tracker.total_trackers_finished}, {video_tracker.get_fps()}fps (Sky360)"
        print(msg)
        frame_output = self.visualise_frame(video_tracker)
        if frame_output is not None:
            self.display(frame_output)
