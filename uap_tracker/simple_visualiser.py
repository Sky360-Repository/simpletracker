import cv2
import uap_tracker.utils as utils
from visualizer import Visualizer


class SimpleVisualiser(Visualizer):
        
    def visualise_frame(self, video_tracker):
        frame_output = video_tracker.get_annotated_image(active_trackers_only=False)
        fps = video_tracker.get_fps()
        utils.stamp_output_frame(video_tracker, frame_output, self.font_size, self.font_colour, fps)
        return frame_output
