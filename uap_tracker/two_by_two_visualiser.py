import cv2
import numpy as np
import uap_tracker.utils as utils


class TwoByTwoVisualiser():

    def __init__(self):
        self.font_size = None
        self.font_colour = None

    def initialise(self, font_size, font_colour):
        self.font_size = font_size
        self.font_colour = font_colour

    def visualise_frame(self, video_tracker, frames, frame_output, key_points, fps):
        frame_input = frames['original']
        frame_masked_background = frames['masked_background']
        utils.stamp_original_frame(frame_input,self.font_size, self.font_colour)

        utils.stamp_output_frame(video_tracker, frame_output, self.font_size, self.font_colour, fps)

        # Create a copy as we need to put text on it and also turn it into a 24 bit image
        frame_masked_background_copy = cv2.cvtColor(
            frame_masked_background.copy(), cv2.COLOR_GRAY2BGR)

        frame_masked_background_with_key_points = cv2.drawKeypoints(
            frame_masked_background_copy, key_points, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        msg = f"Detected {len(key_points)} Key Points (Sky360)"
        cv2.putText(frame_masked_background_with_key_points, msg, (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)

        cv2.putText(frame_masked_background_copy, "Masked Background (Sky360)",
                    (100, 200), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)
        bottom_left_frame = frame_masked_background_copy
        bottom_right_frame = frame_masked_background_with_key_points

        return utils.combine_frames_2x2(
            frame_input, frame_output, bottom_left_frame, bottom_right_frame
        )
