import cv2
import numpy as np
import uap_tracker.utils as utils

class FrameProcessor():

    def __init__(self, frame, dof):
        self.frame = frame
        #self.frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.dof = dof

    @staticmethod
    def CPU(frame_grey, dof):
        return CpuFrameProcessor(frame_grey, dof)

    @staticmethod
    def GPU(frame_grey, dof):
        return GpuFrameProcessor(frame_grey, dof)

    def apply_fisheye_mask(self, frame, mask_pct):
        mask_height = (100 - mask_pct) / 100.0
        mask_radius = mask_height / 2.0
        shape = frame.shape[:2]
        height = shape[0]
        width = shape[1]
        new_width = int(mask_height * width)
        new_height = int(mask_height * height)
        mask = np.zeros(shape, dtype="uint8")
        cv2.circle(mask, (int(width / 2), int(height / 2)),
                   int(min(height, width) * mask_radius), 255, -1)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        clipped_masked_frame = utils.clip_at_center(
            masked_frame,
            (int(width / 2), int(height / 2)),
            width,
            height,
            new_width,
            new_height)
        return clipped_masked_frame

    #def process_optical_flow(self, frame_grey, frame_w, frame_h):
    #    pass

    def resize_frame(self, frame, w, h):
        pass

    #def keypoints_from_bg_subtraction(self, background_subtractor, detection_sensitivity, frame_grey, stream):
    #    pass

    def noise_reduction(self, frame, blur_radius):
        pass

    def convert_to_grey(self, frame):
        pass

class CpuFrameProcessor(FrameProcessor):

    def __init__(self, frame_grey, dof):
        super().__init__(frame_grey, dof)

    def __enter__(self):
        #print('CPU.__enter__')
        return self

    def __exit__(self, type, value, traceback):
        pass
        #print('CPU.__exit__')

    #def process_optical_flow(self, frame_grey, frame_w, frame_h):
    #    # Overload this for a CPU specific implementation
    #    print('CPU.process_optical_flow')
    #    dof_frame = self.dof.process_grey_frame(frame_grey)
    #    return self.resize_frame(dof_frame, frame_w, frame_h)

    def resize_frame(self, frame, w, h):
        # Overload this for a CPU specific implementation
        #print('CPU.resize_frame')
        return cv2.resize(frame, (w, h))

    #def keypoints_from_bg_subtraction(self, background_subtractor, detection_sensitivity, frame_grey, stream):
    #    # Overload this for a CPU specific implementation
    #    print('CPU.keypoints_from_bg_subtraction')
    #    # MG: This needs to be done on an 8 bit grey scale image, the colour image is causing a detection cluster
    #    #frame_masked_background = utils.apply_background_subtraction(frame_grey, background_subtractor)
    #    foreground_mask = background_subtractor.apply(frame_grey)
    #    frame_masked_background = cv2.bitwise_and(frame_grey, frame_grey, mask=foreground_mask)
    #    # Detect new objects of interest to pass to tracker
    #    key_points = utils.perform_blob_detection(frame_masked_background, detection_sensitivity)
    #    return key_points, frame_masked_background

    def noise_reduction(self, frame, blur_radius):
        # Overload this for a CPU specific implementation
        #print('CPU.noise_reduction')
        noise_reduced_frame = cv2.GaussianBlur(frame, (blur_radius, blur_radius), 0)
        # frame_grey = cv2.medianBlur(frame_grey, blur_radius)
        return noise_reduced_frame

    def convert_to_grey(self, frame):
        # Overload this for a CPU specific implementation
        #print('CPU.convert_to_grey')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

class GpuFrameProcessor(FrameProcessor):

    def __init__(self, frame, dof):
        super().__init__(frame, dof)
        #self.gpu_frame_grey = cv2.cuda_GpuMat()

    def __enter__(self):
        #self.gpu_frame_grey.upload(self.frame_grey)
        #print('GPU.__enter__')
        return self

    def __exit__(self, type, value, traceback):
        pass
        #print('GPU.__exit__')

    #def process_optical_flow(self, gpu_frame_grey, frame_w, frame_h):
    #    # Overload this for a GPU specific implementation
    #    print('GPU.process_optical_flow')
    #    gpu_dof_frame = self.dof.process_grey_frame(gpu_frame_grey)
    #    gpu_dof_frame = self.resize_frame(gpu_dof_frame, frame_w, frame_h)
    #    return gpu_dof_frame

    def resize_frame(self, gpu_frame, w, h):
        # Overload this for a GPU specific implementation
        #print('GPU.resize_frame')
        return cv2.cuda.resize(gpu_frame, (w, h))

    #def keypoints_from_bg_subtraction(self, background_subtractor, detection_sensitivity, gpu_frame_grey, stream):
    #    # Overload this for a GPU specific implementation
    #    print('GPU.keypoints_from_bg_subtraction')
    #    #gpu_frame_masked_background = utils.apply_background_subtraction_cuda(gpu_frame_grey, background_subtractor, stream)
    #    gpu_foreground_mask = background_subtractor.apply(gpu_frame_grey, learningRate=0.05, stream=stream)
    #    gpu_frame_masked_background = cv2.cuda.bitwise_and(gpu_frame_grey, gpu_frame_grey, mask=gpu_foreground_mask)
    #    frame_masked_background = gpu_frame_masked_background.download()
    #    # Detect new objects of interest to pass to tracker
    #    key_points = utils.perform_blob_detection(frame_masked_background, detection_sensitivity)
    #    return key_points, frame_masked_background

    def noise_reduction(self, gpu_frame, blur_radius):
        # Overload this for a GPU specific implementation
        #print('GPU.noise_reduction')
        gpuFilter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (blur_radius, blur_radius), 0)
        gpu_noise_reduced_frame = cv2.cuda_Filter.apply(gpuFilter, gpu_frame)
        return gpu_noise_reduced_frame

    def convert_to_grey(self, gpu_frame):
        # Overload this for a GPU specific implementation
        #print('GPU.convert_to_grey')
        return cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
