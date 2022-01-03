import cv2
import numpy as np
from threading import Thread
import uap_tracker.utils as utils
from uap_tracker.mask import FisheyeMask

class FrameProcessor():

    def __init__(self, dense_optical_flow, background_subtractor, resize_frame, resize_dim, noise_reduction, mask_pct, detection_mode, detection_sensitivity, blur_radius):
        self.dense_optical_flow = dense_optical_flow
        self.background_subtractor = background_subtractor
        self.resize_frame = resize_frame
        self.resize_dim = (resize_dim, resize_dim)
        self.noise_reduction = noise_reduction
        self.mask_pct = mask_pct
        self.detection_mode = detection_mode
        self.detection_sensitivity = detection_sensitivity
        self.blur_radius = blur_radius
        self.original_frame_w = 0
        self.original_frame_h = 0
        self.mask = FisheyeMask(mask_pct)

    @staticmethod
    def Select(enable_cuda, dense_optical_flow, background_subtractor, resize_frame, resize_dim, noise_reduction, mask_pct, detection_mode, detection_sensitivity):
        if enable_cuda:
            return FrameProcessor.GPU(dense_optical_flow, background_subtractor, resize_frame, resize_dim, noise_reduction, mask_pct, detection_mode, detection_sensitivity)

        return FrameProcessor.CPU(dense_optical_flow, background_subtractor, resize_frame, resize_dim, noise_reduction, mask_pct, detection_mode, detection_sensitivity)

    @staticmethod
    def CPU(dense_optical_flow, background_subtractor, resize_frame, resize_dim, noise_reduction, mask_pct, detection_mode, detection_sensitivity):
        return CpuFrameProcessor(
            dense_optical_flow,
            background_subtractor,
            resize_frame,
            resize_dim,
            noise_reduction,
            mask_pct,
            detection_mode,
            detection_sensitivity,
            blur_radius=3)

    @staticmethod
    def GPU(dense_optical_flow, background_subtractor, resize_frame, resize_dim, noise_reduction, mask_pct, detection_mode, detection_sensitivity):
        return GpuFrameProcessor(
            dense_optical_flow,
            background_subtractor,
            resize_frame,
            resize_dim,
            noise_reduction,
            mask_pct,
            detection_mode,
            detection_sensitivity,
            blur_radius=3)

    def initialise(self, init_frame):

        shape = self.mask.initialise(init_frame)

        self.original_frame_w = shape[0]
        self.original_frame_h = shape[1]

        if self.resize_frame:
            scale, scaled_width, scaled_height = utils.calc_image_scale(self.original_frame_w, self.original_frame_h, self.resize_dim[0], self.resize_dim[1])
            self.resize_frame = scale
            self.resize_dim = (scaled_width, scaled_height)

            return self.resize_dim

        return shape[:2]

    def resize(self, frame, w, h, stream):
        pass

    def reduce_noise(self, frame, blur_radius, stream):
        pass

    def convert_to_grey(self, frame, stream):
        pass

    def keypoints_from_bg_subtraction(self, frame_grey, stream):
        pass

    def process_optical_flow(self, frame_grey, frame_w, frame_h, stream):
        pass

    def process_frame(self, video_tracker, frame, frame_count, fps, stream):
        pass

class CpuFrameProcessor(FrameProcessor):

    def __init__(self, dense_optical_flow, background_subtractor, resize_frame, resize_dim, noise_reduction, mask_pct, detection_mode, detection_sensitivity, blur_radius):
        super().__init__(dense_optical_flow, background_subtractor, resize_frame, resize_dim, noise_reduction, mask_pct, detection_mode, detection_sensitivity, blur_radius)

    def __enter__(self):
        #print('CPU.__enter__')
        return self

    def __exit__(self, type, value, traceback):
        pass
        #print('CPU.__exit__')

    def resize(self, frame, w, h, stream):
        # Overload this for a CPU specific implementation
        #print('CPU.resize_frame')
        return cv2.resize(frame, (w, h))

    def reduce_noise(self, frame, blur_radius, stream):
        # Overload this for a CPU specific implementation
        #print('CPU.noise_reduction')
        noise_reduced_frame = cv2.GaussianBlur(frame, (blur_radius, blur_radius), 0)
        # frame_grey = cv2.medianBlur(frame_grey, blur_radius)
        return noise_reduced_frame

    def convert_to_grey(self, frame, stream):
        # Overload this for a CPU specific implementation
        #print('CPU.convert_to_grey')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def keypoints_from_bg_subtraction(self, frame_grey, stream):
        # Overload this for a CPU specific implementation
        #print('CPU.keypoints_from_bg_subtraction')
        # MG: This needs to be done on an 8 bit grey scale image, the colour image is causing a detection cluster
        foreground_mask = self.background_subtractor.apply(frame_grey)
        frame_masked_background = cv2.bitwise_and(frame_grey, frame_grey, mask=foreground_mask)
        # Detect new objects of interest to pass to tracker
        key_points = utils.perform_blob_detection(frame_masked_background, self.detection_sensitivity)
        return key_points, frame_masked_background

    def process_optical_flow(self, frame_grey, frame_w, frame_h, stream):
        # Overload this for a CPU specific implementation
        # print('CPU.process_optical_flow')
        dof_frame = self.dense_optical_flow.process_grey_frame(frame_grey)
        return self.resize(dof_frame, frame_w, frame_h, stream)

    def process_frame(self, video_tracker, frame, frame_count, fps, stream=None):

        # print(f" fps:{int(fps)}", end='\r')

        worker_threads = []
        bboxes = []
        keypoints = []

        frame = self.mask.apply(frame)

        # Mike: As part of the initialisation method we worked out that the frame needs to be resized
        if self.resize_frame:
            frame = self.resize(frame, self.resize_dim[0], self.resize_dim[1], stream)

        frame_grey = self.convert_to_grey(frame, stream)

        if self.noise_reduction:
            frame_grey = self.reduce_noise(frame_grey, self.blur_radius, stream)

        video_tracker.add_image(video_tracker.FRAME_TYPE_ORIGINAL, frame)
        video_tracker.add_image(video_tracker.FRAME_TYPE_GREY, frame_grey)

        if self.detection_mode == 'background_subtraction':

            keypoints, frame_masked_background = self.keypoints_from_bg_subtraction(frame_grey, stream)
            video_tracker.add_image(video_tracker.FRAME_TYPE_MASKED_BACKGROUND, frame_masked_background)

            if frame_count < 5:
                # Need 5 frames to get the background subtractor initialised
                return keypoints

            bboxes = [utils.kp_to_bbox(x) for x in keypoints]

            if self.dense_optical_flow is not None:
                optical_flow_thread = Thread(target=self.perform_optical_flow_task,
                                             args=(video_tracker, frame_count, frame_grey, self.resize_dim[0], self.resize_dim[1], stream))
                optical_flow_thread.start()
                worker_threads.append(optical_flow_thread)

        video_tracker.update_trackers(video_tracker.tracker_type, bboxes, frame)

        frame_count + 1

        # Mike: Wait for worker threads to join before publishing events as their results might be required
        for worker_thread in worker_threads:
            worker_thread.join()

        return keypoints

    def perform_optical_flow_task(self, video_tracker, frame_count, frame_grey, frame_w, frame_h, stream):
        optical_flow_frame = self.process_optical_flow(frame_grey, frame_w, frame_h, stream)
        video_tracker.add_image(video_tracker.FRAME_TYPE_OPTICAL_FLOW, optical_flow_frame)

class GpuFrameProcessor(FrameProcessor):

    def __init__(self, dense_optical_flow, background_subtractor, resize_frame, resize_dim, noise_reduction, mask_pct, detection_mode, detection_sensitivity, blur_radius):
        super().__init__(dense_optical_flow, background_subtractor, resize_frame, resize_dim, noise_reduction, mask_pct, detection_mode, detection_sensitivity, blur_radius)

    def __enter__(self):
        #print('GPU.__enter__')
        return self

    def __exit__(self, type, value, traceback):
        pass
        #print('GPU.__exit__')

    def resize(self, gpu_frame, w, h, stream):
        # Overload this for a GPU specific implementation
        #print('GPU.resize_frame')
        return cv2.cuda.resize(gpu_frame, (w, h), stream=stream)

    def reduce_noise(self, gpu_frame, blur_radius, stream):
        # Overload this for a GPU specific implementation
        #print('GPU.noise_reduction')
        gpuFilter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (blur_radius, blur_radius), 0)
        gpu_noise_reduced_frame = cv2.cuda_Filter.apply(gpuFilter, gpu_frame, stream=stream)
        return gpu_noise_reduced_frame

    def convert_to_grey(self, gpu_frame, stream):
        # Overload this for a GPU specific implementation
        #print('GPU.convert_to_grey')
        return cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY, stream=stream)

    def keypoints_from_bg_subtraction(self, gpu_frame_grey, stream):
        # Overload this for a GPU specific implementation
        # print('GPU.keypoints_from_bg_subtraction')
        gpu_foreground_mask = self.background_subtractor.apply(gpu_frame_grey, learningRate=0.05, stream=stream)
        gpu_frame_masked_background = cv2.cuda.bitwise_and(gpu_frame_grey, gpu_frame_grey, mask=gpu_foreground_mask, stream=stream)
        frame_masked_background = gpu_frame_masked_background.download()
        # Detect new objects of interest to pass to tracker
        key_points = utils.perform_blob_detection(frame_masked_background, self.detection_sensitivity)
        return key_points, frame_masked_background

    def process_optical_flow(self, gpu_frame_grey, frame_w, frame_h, stream):
        # Overload this for a GPU specific implementation
        #print('GPU.process_optical_flow')
        gpu_dof_frame = self.dense_optical_flow.process_grey_frame(gpu_frame_grey, stream)
        gpu_dof_frame = self.resize(gpu_dof_frame, frame_w, frame_h, stream)
        return gpu_dof_frame

    def process_frame(self, video_tracker, frame, frame_count, fps, stream):

         # print(f" fps:{int(fps)}", end='\r')

         worker_threads = []
         bboxes = []
         keypoints = []

         # Mike: Need to figure out how to offload to CUDA
         frame = self.mask.apply(frame)

         gpu_frame = cv2.cuda_GpuMat()
         gpu_frame.upload(frame, stream=stream)

         # Mike: As part of the initialisation method we worked out that the frame needs to be resized
         if self.resize_frame:
             frame = self.resize(gpu_frame, self.resize_dim[0], self.resize_dim[1], stream)

         gpu_frame_grey = self.convert_to_grey(gpu_frame, stream)

         if self.noise_reduction:
             gpu_frame_grey = self.reduce_noise(gpu_frame_grey, self.blur_radius, stream)

         frame = gpu_frame.download()
         frame_grey = gpu_frame_grey.download()

         video_tracker.add_image(video_tracker.FRAME_TYPE_ORIGINAL, frame)
         video_tracker.add_image(video_tracker.FRAME_TYPE_GREY, frame_grey)

         if self.detection_mode == 'background_subtraction':

             keypoints, frame_masked_background = self.keypoints_from_bg_subtraction(gpu_frame_grey, stream)
             video_tracker.add_image(video_tracker.FRAME_TYPE_MASKED_BACKGROUND, frame_masked_background)

             if frame_count < 5:
                 # Need 5 frames to get the background subtractor initialised
                 return keypoints

             bboxes = [utils.kp_to_bbox(x) for x in keypoints]

             if self.dense_optical_flow is not None:
                 #self.perform_optical_flow_task(video_tracker, frame_count, gpu_frame_grey, self.resize_dim[0], self.resize_dim[1], stream)
                 optical_flow_thread = Thread(target=self.perform_optical_flow_task,
                                              args=(video_tracker, frame_count, gpu_frame_grey, self.resize_dim[0], self.resize_dim[1], stream))
                 optical_flow_thread.start()
                 worker_threads.append(optical_flow_thread)

         video_tracker.update_trackers(video_tracker.tracker_type, bboxes, frame)

         frame_count + 1

         # Mike: Wait for worker threads to join before publishing events as their results might be required
         for worker_thread in worker_threads:
             worker_thread.join()

         return keypoints

    def perform_optical_flow_task(self, video_tracker, frame_count, gpu_frame_grey, frame_w, frame_h, stream):
        gpu_dof_frame = self.process_optical_flow(gpu_frame_grey, frame_w, frame_h, stream)
        dof_frame = gpu_dof_frame.download()
        video_tracker.add_image(video_tracker.FRAME_TYPE_OPTICAL_FLOW, dof_frame)