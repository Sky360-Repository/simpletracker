import cv2
import numpy as np

class FrameProcessor():

    def __init__(self, frame):
        this.frame = frame

    @staticmethod
    def CPU(frame):
        return CpuFrameProcessor(frame)

    @staticmethod
    def GPU(frame):
        return GpuFrameProcessor(frame)

    def process_optical_flow(self, frame_grey, frame_w, frame_h):
        pass

    def resize_frame(self, frame, w, h):
        pass

    def keypoints_from_bg_subtraction(self, frame_grey, stream):
        pass

    def noise_reduction(self, frame, blur_radius):
        pass

    def convert_to_grey(self, frame):
        pass

class CpuFrameProcessor(FrameProcessor):

    def __init__(self, frame):
        super().__init__(frame)

    def __enter__(self):
        print('CPU.__enter__')
        return self

    def __exit__(self, type, value, traceback):
        print('CPU.__exit__')

    def process_optical_flow(self, frame_grey, frame_w, frame_h):
        # Overload this for a CPU specific implementation
        print('CPU.process_optical_flow')

    def resize_frame(self, frame, w, h):
        # Overload this for a CPU specific implementation
        print('CPU.resize_frame')

    def keypoints_from_bg_subtraction(self, frame_grey, stream):
        # Overload this for a CPU specific implementation
        print('CPU.keypoints_from_bg_subtraction')

    def noise_reduction(self, frame, blur_radius):
        # Overload this for a CPU specific implementation
        print('CPU.noise_reduction')

    def convert_to_grey(self, frame):
        # Overload this for a CPU specific implementation
        print('CPU.convert_to_grey')

class GpuFrameProcessor(FrameProcessor):

    def __init__(self, frame):
        super().__init__(frame)
        this.gpu_frame = cv2.cuda_GpuMat()

    def __enter__(self):
        this.gpu_frame.upload(this.frame)
        print('GPU.__enter__')
        return self

    def __exit__(self, type, value, traceback):
        print('GPU.__exit__')

    def process_optical_flow(self, frame_grey, frame_w, frame_h):
        # Overload this for a GPU specific implementation
        print('GPU.process_optical_flow')

    def resize_frame(self, frame, w, h):
        # Overload this for a GPU specific implementation
        print('GPU.resize_frame')

    def keypoints_from_bg_subtraction(self, frame_grey, stream):
        # Overload this for a GPU specific implementation
        print('GPU.keypoints_from_bg_subtraction')

    def noise_reduction(self, frame, blur_radius):
        # Overload this for a GPU specific implementation
        print('GPU.noise_reduction')

    def convert_to_grey(self, frame):
        # Overload this for a GPU specific implementation
        print('GPU.convert_to_grey')