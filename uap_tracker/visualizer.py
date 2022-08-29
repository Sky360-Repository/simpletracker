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
from uap_tracker import utils

##############################################################################################################################
# Base class for various visualiser implementations. Viualisers are generally there as the display part of the application.  #
# Essentially its the GUI part of the application.                                                                           #
##############################################################################################################################
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

####################################################################################################################
# No op visualiser has no GUI part. It basically just echos out the status of the application to the command line. #
####################################################################################################################
class NoOpVisualiser(Visualizer):

    def visualise_frame(self, video_tracker):
        return None

######################################################
# The Simple visualiser displays the annotated frame #
######################################################
class SimpleVisualiser(Visualizer):
        
    def visualise_frame(self, video_tracker):
        frame_output = video_tracker.get_annotated_image(active_trackers_only=False)
        fps = video_tracker.get_fps()
        utils.stamp_output_frame(video_tracker, frame_output, self.font_size, self.font_colour, fps, self.font_thickness)
        return frame_output

###########################################################
# The TwoByTwo visualiser displays four frames:           #
#   top left: Original Frame                              #
#   top right: Annotated Frame                            #
#   bottom left: Masked Background Frame                  #
#   bottom right: Masked Background With Key Points Frame #
###########################################################
class TwoByTwoVisualiser(Visualizer):

    def visualise_frame(self, video_tracker):
        frame_input = video_tracker.get_image(video_tracker.FRAME_TYPE_ORIGINAL)
        frame_masked_background = video_tracker.get_image(video_tracker.FRAME_TYPE_MASKED_BACKGROUND)
        frame_output = video_tracker.get_annotated_image(active_trackers_only=False)

        key_points = video_tracker.get_keypoints()
        fps = video_tracker.get_fps()

        utils.stamp_original_frame(
            frame_input, self.font_size, self.font_colour, self.font_thickness)

        utils.stamp_output_frame(
            video_tracker, frame_output, self.font_size, self.font_colour, fps, self.font_thickness)

        # Create a copy as we need to put text on it and also turn it into a 24 bit image
        frame_masked_background_copy = cv2.cvtColor(frame_masked_background, cv2.COLOR_GRAY2BGR)

        frame_masked_background_with_key_points = cv2.drawKeypoints(
            frame_masked_background_copy, key_points, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        msg = f"Detected {len(key_points)} Key Points (Sky360)"
        cv2.putText(frame_masked_background_with_key_points, msg, (25, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, self.font_thickness)

        cv2.putText(frame_masked_background_copy, "Masked Background (Sky360)",
                    (25, 50), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, self.font_thickness)
        bottom_left_frame = frame_masked_background_copy
        bottom_right_frame = frame_masked_background_with_key_points

        return utils.combine_frames_2x2(frame_input, frame_output, bottom_left_frame, bottom_right_frame)

#################################################
# The TwoByTwo visualiser displays four frames: #
#   top left: Original Frame                    #
#   top right: Annotated Frame                  #
#   bottom left: Optical Flow Frame             #
#   bottom right: Optical Flow Frame            #
#################################################
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