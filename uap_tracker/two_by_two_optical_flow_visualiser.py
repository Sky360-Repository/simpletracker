import cv2
import numpy as np
import uap_tracker.utils as utils


class TwoByTwoOpticalFlowVisualiser():

    def __init__(self):
        self.font_size = None
        self.font_colour = None

    def initialise(self, font_size, font_colour):
        self.font_size = font_size
        self.font_colour = font_colour

    def visualise_frame(self, video_tracker, frame_input, frame_masked_background, frame_output, optical_flow_frame, key_points, fps):

        utils.stamp_original_frame(
            frame_input, self.font_size, self.font_colour)

        utils.stamp_output_frame(
            video_tracker, frame_output, self.font_size, self.font_colour, fps)

        bottom_left_frame = optical_flow_frame
        bottom_right_frame = optical_flow_frame

        return utils.combine_frames_2x2(
            frame_input, frame_output, bottom_left_frame, bottom_right_frame
        )
