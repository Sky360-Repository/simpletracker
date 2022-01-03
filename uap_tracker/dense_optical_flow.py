import cv2
import numpy as np
import uap_tracker.utils as utils

class DenseOpticalFlow():

    def __init__(self, width, height):
        self.width = width
        self.height = height

    @staticmethod
    def Select(enable_cuda, width, height):
        if enable_cuda:
            return DenseOpticalFlow.GPU(width, height)

        return DenseOpticalFlow.CPU(width, height)

    @staticmethod
    def CPU(width, height):
        return CpuDenseOpticalFlow(width, height)

    @staticmethod
    def GPU(width, height):
        return GpuDenseOpticalFlow(width, height)

    def process_grey_frame(self, frame, stream):
        pass

class CpuDenseOpticalFlow(DenseOpticalFlow):

    def __init__(self, width, height):
        super().__init__(width, height)

        self.previous_frame = None

        # create hsv output for optical flow
        self.hsv = np.zeros((self.height, self.width, 3), np.float32)

        # set saturation to 1
        self.hsv[..., 1] = 1.0

    def _resize(self, frame):
        return cv2.resize(frame, (self.width, self.height))

    def process_grey_frame(self, frame, stream=None):
        frame = self._resize(frame)
        if self.previous_frame is None:
            # First time round, save frame and return blank image
            self.previous_frame = frame
            return np.zeros((self.height, self.width, 3), np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            self.previous_frame, frame, None, 0.5, 5, 15, 3, 5, 1.2, 0,
        )

        # convert from cartesian to polar coordinates to get magnitude and angle
        magnitude, angle = cv2.cartToPolar(
            flow[..., 0], flow[..., 1], angleInDegrees=True,
        )

        # set hue according to the angle of optical flow
        self.hsv[..., 0] = angle * ((1 / 360.0) * (180 / 255.0))

        # set value according to the normalized magnitude of optical flow
        self.hsv[..., 2] = magnitude
        #cv2.normalize(
        #    magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, -1,
        #)

        # multiply each pixel value to 255
        hsv_8u = np.uint8(self.hsv * 255.0)

        # convert hsv to bgr
        bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

        self.previous_frame = frame

        return bgr

class GpuDenseOpticalFlow(DenseOpticalFlow):

    def __init__(self, width, height):
        super().__init__(width, height)
        self.previous_gpu_frame = None

    def _resize(self, gpu_frame, stream):
        return cv2.cuda.resize(gpu_frame, (self.width, self.height), stream=stream)

    def process_grey_frame(self, gpu_frame, stream):

        gpu_frame = self._resize(gpu_frame, stream)

        gpu_bgr_frame = cv2.cuda_GpuMat()
        if self.previous_gpu_frame is None:
            # First time round, save frame and return blank image
            self.previous_gpu_frame = gpu_frame

            # create gpu_hsv output for optical flow
            self.gpu_hsv_frame = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC3)
            self.gpu_hsv_8u_frame = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_8UC3)

            # channels
            self.gpu_h_channel = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
            self.gpu_s_channel = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
            self.gpu_v_channel = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)

            # set saturation to 1
            self.gpu_s_channel.upload(np.ones_like(gpu_frame.download(), np.float32), stream=stream)

            gpu_bgr_frame.upload(np.zeros((self.height, self.width, 3), np.uint8), stream=stream)

            return gpu_bgr_frame

        # create optical flow instance
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)

        # calculate optical flow
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(gpu_flow, self.previous_gpu_frame, gpu_frame, None, stream=stream)

        gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y], stream=stream)

        # convert from cartesian to polar coordinates to get magnitude and angle
        gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(gpu_flow_x, gpu_flow_y, angleInDegrees=True, stream=stream)

        # set value to normalized magnitude from 0 to 1
        self.gpu_v_channel = cv2.cuda.normalize(gpu_magnitude, 0.0, 1.0, cv2.NORM_MINMAX, -1, stream=stream)

        # get angle of optical flow
        angle = gpu_angle.download()
        angle *= (1 / 360.0) * (180 / 255.0)

        # set hue according to the angle of optical flow
        self.gpu_h_channel.upload(angle, stream=stream)

        # merge h,s,v channels
        cv2.cuda.merge([self.gpu_h_channel, self.gpu_s_channel, self.gpu_v_channel], self.gpu_hsv_frame, stream=stream)

        # multiply each pixel value to 255
        self.gpu_hsv_frame.convertTo(cv2.CV_8U, 255.0, self.gpu_hsv_8u_frame, 0.0)

        # convert hsv to bgr
        gpu_bgr_frame = cv2.cuda.cvtColor(self.gpu_hsv_8u_frame, cv2.COLOR_HSV2BGR, stream=stream)

        self.previous_gpu_frame = gpu_frame

        return gpu_bgr_frame
