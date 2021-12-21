import cv2
import numpy as np
from threading import Thread
from uap_tracker.stopwatch import Stopwatch
#import multiprocessing
import uap_tracker.utils as utils
from uap_tracker.tracker import Tracker
from uap_tracker.background_subtractor_factory import BackgroundSubtractorFactory
from uap_tracker.dense_optical_flow import CpuDenseOpticalFlow, GpuDenseOpticalFlow
from uap_tracker.frame_processor import  FrameProcessor, CpuFrameProcessor, GpuFrameProcessor

#
# Tracks multiple objects in a video
#


class VideoTracker():

    DETECTION_SENSITIVITY_HIGH = 1
    DETECTION_SENSITIVITY_NORMAL = 2
    DETECTION_SENSITIVITY_LOW = 3

    def __init__(self, detection_mode, events, visualizer, detection_sensitivity=2, mask_pct=8, noise_reduction=True, resize_frame=True,
                 calculate_optical_flow=True, max_active_trackers=10, tracker_type='CSRT', background_subtractor_type='KNN'):

        print(
            f"Initializing Tracker:\n  resize_frame:{resize_frame}\n  noise_reduction: {noise_reduction}\n  mask_pct:{mask_pct}\n  sensitivity:{detection_sensitivity}\n  max_active_trackers:{max_active_trackers}\n  tracker_type:{tracker_type}\n  background_subtractor_type:{background_subtractor_type}")

        self.detection_mode = detection_mode
        if detection_sensitivity < 1 or detection_sensitivity > 3:
            raise Exception(
                f"Unknown sensitivity option ({detection_sensitivity}). 1, 2 and 3 is supported not {detection_sensitivity}.")

        self.detection_sensitivity = detection_sensitivity
        self.total_trackers_finished = 0
        self.total_trackers_started = 0
        self.live_trackers = []
        self.events = events
        self.visualizer = visualizer
        self.normalised_w_h = (1024, 1024)
        self.blur_radius = 3
        self.max_active_trackers = max_active_trackers
        self.mask_pct = mask_pct
        self.calculate_optical_flow = calculate_optical_flow

        self.noise_reduction = noise_reduction
        self.resize_frame = resize_frame
        self.tracker_type = None
        self.background_subtractor_type = None
        self.background_subtractor = None
        self.frame_output = None
        self.frame_masked_background = None

        self.dof = CpuDenseOpticalFlow(480, 480)
        self.dof_cuda = GpuDenseOpticalFlow(480, 480)

        self.tracker_type = tracker_type
        self.background_subtractor_type = background_subtractor_type

        self.background_subtractor = BackgroundSubtractorFactory.create(
            self.background_subtractor_type, self.detection_sensitivity)

    @property
    def is_tracking(self):
        return len(self.live_trackers) > 0

    def active_trackers(self):
        trackers = filter(lambda x: x.is_tracking(), self.live_trackers)
        if trackers is None:
            return []
        else:
            return trackers

    def create_trackers_from_keypoints(self, tracker_type, key_points, frame):
        for kp in key_points:
            bbox = utils.kp_to_bbox(kp)
            # print(bbox)

            # Initialize tracker with first frame and bounding box
            if not utils.is_bbox_being_tracked(self.live_trackers, bbox):
                self.create_and_add_tracker(tracker_type, frame, bbox)

    def create_and_add_tracker(self, tracker_type, frame, bbox):
        if not bbox:
            raise Exception("null bbox")

        self.total_trackers_started += 1

        tracker = Tracker(self.total_trackers_started, tracker_type,
                          frame, bbox)
        tracker.update(frame)
        self.live_trackers.append(tracker)

    def update_trackers(self, tracker_type, bboxes, frame):

        unmatched_bboxes = bboxes.copy()
        failed_trackers = []
        tracker_count = len(self.live_trackers)

        #task_args = []
        #results = []

        #for i in range(tracker_count):
        #    task_args.append((i, frame, results))

        # Mike: We can do the tracker updates in parallel
        #with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        #    pool.starmap(self.update_tracker_task_2, task_args)

        threads = [None] * tracker_count
        results = [None] * tracker_count

        # Mike: We can do the tracker updates in parallel
        for i in range(tracker_count):
            tracker = self.live_trackers[i]
            threads[i] = Thread(target=self.update_tracker_task, args=(tracker, frame, results, i))
            threads[i].start()

        # Mike: We got to wait for the threads to join before we proceed
        for i in range(tracker_count):
            threads[i].join()

        for i in range(tracker_count):
            tracker = self.live_trackers[i]
            ok, bbox = results[i]
            if not ok:
                # Tracking failure
                failed_trackers.append(tracker)

            # Try to match the new detections with this tracker
            #if ok:
            for new_bbox in bboxes:
                if new_bbox in unmatched_bboxes:
                    overlap = utils.bbox_overlap(bbox, new_bbox)
                    # print(f'Overlap: {overlap}; bbox:{bbox}, new_bbox:{new_bbox}')
                    if overlap > 0.2:
                        unmatched_bboxes.remove(new_bbox)

        # remove failed trackers from live tracking
        for tracker in failed_trackers:
            self.live_trackers.remove(tracker)
            self.total_trackers_finished += 1

        # Add new detections to live tracker
        for new_bbox in unmatched_bboxes:
            # Hit max trackers?
            if len(self.live_trackers) < self.max_active_trackers:
                if not utils.is_bbox_being_tracked(self.live_trackers, new_bbox):
                    self.create_and_add_tracker(tracker_type, frame, new_bbox)

    def process_frame(self, frame_proc, frame, frame_count, fps):

        with Stopwatch(mask='Frame '+str(frame_count)+': Took {s:0.4f} seconds to process', quiet=True):
            # print(f" fps:{int(fps)}", end='\r')
            self.fps = fps
            self.frame_count = frame_count
            frame_w = frame.shape[0]
            frame_h = frame.shape[1]
            worker_threads = []

            frame = frame_proc.apply_fisheye_mask(frame, self.mask_pct)

            if self.resize_frame:
                scale, scaled_width, scaled_height = utils.calc_image_scale(frame_w, frame_h, self.normalised_w_h[0], self.normalised_w_h[1])
                if scale:
                    frame = frame_proc.resize_frame(frame, scaled_width, scaled_height)

            frame_grey = frame_proc.convert_to_grey(frame)

            # Mike: Not able to offload to CUDA
            if self.noise_reduction:
                frame_grey = frame_proc.noise_reduction(frame_grey, self.blur_radius)

            self.frames = {
                'original': frame,
                'grey': frame_grey
            }

            if self.detection_mode == 'background_subtraction':

                # Mike: Able to offload some of this to CUDA
                if self.background_subtractor_type == 'MOG2_CUDA':
                    gpu_frame_grey = cv2.cuda_GpuMat()
                    gpu_frame_grey.upload(frame_grey)
                    keypoints, frame_masked_background = self.keypoints_from_bg_subtraction_cuda(gpu_frame_grey, cv2.cuda.Stream_Null())
                else:
                    keypoints, frame_masked_background = self.keypoints_from_bg_subtraction(frame_grey)

                if frame_count < 5:
                    # Need 5 frames to get the background subtractor initialised
                    return

                self.frames['masked_background'] = frame_masked_background
                bboxes = [utils.kp_to_bbox(x) for x in keypoints]

                if self.calculate_optical_flow:
                    # Mike: The optical flow stuff apears to just add the frame to the frames list, so is an ideal candidate
                    # to perform on a different thread
                    optical_flow_thread = Thread(target=self.perform_optical_flow_task,
                                                 args=(frame_count, frame_grey, scaled_width, scaled_height))
                    optical_flow_thread.start()
                    worker_threads.append(optical_flow_thread)
            else:
                bboxes = []
                keypoints = []

            self.keypoints = keypoints

            self.update_trackers(self.tracker_type, bboxes, frame)

            frame_count + 1

            # Mike: Wait for worker threads to join before publishing events as their results might be required
            for worker_thread in worker_threads:
                worker_thread.join()

            if self.events is not None:
                self.events.publish_process_frame(self)

            if self.visualizer is not None:
                self.visualizer.Visualize(self)

    def process_frame_cuda(self, frame_proc, frame, frame_count, fps):

        with Stopwatch(mask='CUDA Frame '+str(frame_count)+': Took {s:0.4f} seconds to process', quiet=True):
            # print(f" fps:{int(fps)}", end='\r')
            self.fps = fps
            self.frame_count = frame_count
            frame_w = frame.shape[0]
            frame_h = frame.shape[1]
            worker_threads = []
            self.frames = {}

            # Mike: Not able to offload to CUDA
            frame = frame_proc.apply_fisheye_mask(frame, self.mask_pct)

            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            if self.resize_frame:
                scale, scaled_width, scaled_height = utils.calc_image_scale(frame_w, frame_h, self.normalised_w_h[0], self.normalised_w_h[1])
                if scale:
                    gpu_frame = frame_proc.resize_frame(gpu_frame, scaled_width, scaled_height)

            gpu_frame_grey = frame_proc.convert_to_grey(gpu_frame)

            if self.noise_reduction:
                gpu_frame_grey = frame_proc.noise_reduction(gpu_frame_grey, self.blur_radius)

            self.frames['original'] = gpu_frame.download()
            frame_grey = gpu_frame_grey.download()
            self.frames['grey'] = frame_grey

            if self.detection_mode == 'background_subtraction':

                # Mike: Able to offload some of this to CUDA
                if self.background_subtractor_type == 'MOG2_CUDA':
                    keypoints, frame_masked_background = self.keypoints_from_bg_subtraction_cuda(gpu_frame_grey, cv2.cuda.Stream_Null())
                else:
                    keypoints, frame_masked_background = self.keypoints_from_bg_subtraction(frame_grey)

                if frame_count < 5:
                    # Need 5 frames to get the background subtractor initialised
                    return

                self.frames['masked_background'] = frame_masked_background
                bboxes = [utils.kp_to_bbox(x) for x in keypoints]

                if self.calculate_optical_flow:
                    # Mike: The optical flow stuff apears to just add the frame to the frames list, so is an ideal candidate
                    # to perform on a different thread
                    optical_flow_cuda_thread = Thread(target=self.perform_optical_flow_cuda_task,
                                                 args=(frame_count, gpu_frame_grey, scaled_width, scaled_height))
                    optical_flow_cuda_thread.start()
                    worker_threads.append(optical_flow_cuda_thread)

            else:
                bboxes = []
                keypoints = []

            self.keypoints = keypoints

            self.update_trackers(self.tracker_type, bboxes, gpu_frame.download())

            frame_count + 1

            # Mike: Wait for worker threads to join before publishing events as their results might be required
            for worker_thread in worker_threads:
                worker_thread.join()

            if self.events is not None:
                self.events.publish_process_frame(self)

            if self.visualizer is not None:
                self.visualizer.Visualize(self)

    def optical_flow(self, frame_grey, frame_w, frame_h):
        dof_frame = self.dof.process_grey_frame(frame_grey)
        return cv2.resize(dof_frame, (frame_w, frame_h))

    def optical_flow_cuda(self, gpu_frame_grey, frame_w, frame_h):
        gpu_dof_frame = self.dof_cuda.process_grey_frame(gpu_frame_grey)
        gpu_dof_frame = cv2.cuda.resize(gpu_dof_frame, (frame_w, frame_h))
        return gpu_dof_frame

    def keypoints_from_bg_subtraction(self, frame_grey):
        # MG: This needs to be done on an 8 bit grey scale image, the colour image is causing a detection cluster
        frame_masked_background = utils.apply_background_subtraction(
            frame_grey, self.background_subtractor)

        # Detect new objects of interest to pass to tracker
        key_points = utils.perform_blob_detection(
            frame_masked_background, self.detection_sensitivity)
        return key_points, frame_masked_background

    def keypoints_from_bg_subtraction_cuda(self, gpu_frame_grey, stream):
        # MG: This needs to be done on an 8 bit grey scale image, the colour image is causing a detection cluster
        gpu_frame_masked_background = utils.apply_background_subtraction_cuda(
            gpu_frame_grey, self.background_subtractor, stream)

        frame_masked_background = gpu_frame_masked_background.download()

        # Detect new objects of interest to pass to tracker
        key_points = utils.perform_blob_detection(
            frame_masked_background, self.detection_sensitivity)
        return key_points, frame_masked_background

    def finalise(self):
        if self.events is not None:
            self.events.publish_finalise(
                self.total_trackers_started, self.total_trackers_finished)

    # called from listeners / visualizers
    # returns named image for current frame
    def get_image(self, frame_name):
        return self.frames[frame_name]

    # called from listeners / visualizers
    # returns annotated image for current frame
    def get_annotated_image(self, active_trackers_only=True):
        annotated_frame = self.frames.get('annotated_image', None)
        if annotated_frame is None:
            annotated_frame = self.frames['original'].copy()
            if active_trackers_only:
                for tracker in self.active_trackers():
                    utils.add_bbox_to_image(tracker.get_bbox(), annotated_frame, tracker.id, 1, tracker.bbox_color())
            else:
                for tracker in self.live_trackers:
                    utils.add_bbox_to_image(tracker.get_bbox(), annotated_frame, tracker.id, 1, tracker.bbox_color())
            self.frames['annotated_image'] = annotated_frame

        return self.frames['annotated_image']

    # called from listeners / visualizers
    # returns all images for current frame

    def get_images(self):
        return self.frames

    def get_fps(self):
        return int(self.fps)

    def get_frame_count(self):
        return self.frame_count

    def get_live_trackers(self):
        return self.live_trackers

    def get_keypoints(self):
        return self.keypoints

    # Mike: Identifying tasks that can be called on seperate threads to try and speed this sucker up
    def update_tracker_task(self, tracker, frame, results, index):
        results[index] = tracker.update(frame)

    def perform_optical_flow_task(self, frame_count, frame_grey, frame_w, frame_h):
        self.frames['optical_flow'] = self.optical_flow(frame_grey, frame_w, frame_h)

    def perform_optical_flow_cuda_task(self, frame_count, gpu_frame_grey, frame_w, frame_h):
        gpu_dof_frame = self.optical_flow_cuda(gpu_frame_grey, frame_w, frame_h)
        self.frames['optical_flow'] = gpu_dof_frame.download()
