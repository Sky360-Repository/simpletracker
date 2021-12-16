import cv2
import numpy as np

class FrameProcessor():

    def __init__(self, video_tracker):
        self.video_tracker = video_tracker

    def process_frame(self, frame, frame_count, fps):
        # this will be the main frame process method and will call into the overloaded methods from the more specialised
        # implimentations
        pass

    def process_optical_flow(self, frame_grey, frame_w, frame_h):
        pass

    def resize_frame(self, frame, w, h):
        pass

    def keypoints_from_bg_subtraction(self, frame_grey, stream):
        pass

    def noise_reduction(self, frame, blur_radius):
        pass


class CpuFrameProcessor(FrameProcessor):

    def __init__(self, video_tracker):
        super().__init__(video_tracker)

    def process_optical_flow(self, frame_grey, frame_w, frame_h):
        # Overload this for a CPU specific implementation
        pass

    def resize_frame(self, frame, w, h):
        # Overload this for a CPU specific implementation
        pass

    def keypoints_from_bg_subtraction(self, frame_grey, stream):
        # Overload this for a CPU specific implementation
        pass

    def noise_reduction(self, frame, blur_radius):
        # Overload this for a CPU specific implementation
        pass

class GpuFrameProcessor(FrameProcessor):

    def __init__(self, video_tracker):
        super().__init__(video_tracker)

    def process_optical_flow(self, frame_grey, frame_w, frame_h):
        # Overload this for a GPU specific implementation
        pass

    def resize_frame(self, frame, w, h):
        # Overload this for a GPU specific implementation
        pass

    def keypoints_from_bg_subtraction(self, frame_grey, stream):
        # Overload this for a GPU specific implementation
        pass

    def noise_reduction(self, frame, blur_radius):
        # Overload this for a GPU specific implementation
        pass
