import cv2
import numpy as np
import uap_tracker.utils as utils
from uap_tracker.visualizer import Visualizer


class TwoByTwoOpticalFlowVisualiser(Visualizer):

    def visualise_frame(self, video_tracker):

        frame_input = video_tracker.get_image('original')
        optical_flow_frame = video_tracker.get_image('optical_flow')
        frame_output = frame_input.copy()
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
