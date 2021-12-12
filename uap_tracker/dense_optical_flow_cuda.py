import cv2
import numpy as np
import uap_tracker.utils as utils


class DenseOpticalFlowCuda():

    def __init__(self, width, height):
        self.previous_gpu_frame = None
        self.width = width
        self.height = height

    def _resize_cuda(self, gpu_frame):
        return cv2.cuda.resize(gpu_frame, (self.width, self.height))

    def process_grey_frame(self, gpu_frame):
        gpu_frame = self._resize_cuda(gpu_frame)

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
            self.gpu_s_channel.upload(np.ones_like(gpu_frame.download(), np.float32))

            gpu_bgr_frame.upload(np.zeros((self.height, self.width), np.uint8))

            return gpu_bgr_frame

        # create optical flow instance
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)

        # calculate optical flow
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(gpu_flow, self.previous_gpu_frame, gpu_frame, None)

        gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

        # convert from cartesian to polar coordinates to get magnitude and angle
        gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(gpu_flow_x, gpu_flow_y, angleInDegrees=True)

        # set value to normalized magnitude from 0 to 1
        self.gpu_v_channel = cv2.cuda.normalize(gpu_magnitude, 0.0, 1.0, cv2.NORM_MINMAX, -1)

        # get angle of optical flow
        angle = gpu_angle.download()
        angle *= (1 / 360.0) * (180 / 255.0)

        # set hue according to the angle of optical flow
        self.gpu_h_channel.upload(angle)

        # merge h,s,v channels
        cv2.cuda.merge([self.gpu_h_channel, self.gpu_s_channel, self.gpu_v_channel], self.gpu_hsv_frame)

        # multiply each pixel value to 255
        self.gpu_hsv_frame.convertTo(cv2.CV_8U, 255.0, self.gpu_hsv_8u_frame, 0.0)

        # convert hsv to bgr
        gpu_bgr_frame = cv2.cuda.cvtColor(self.gpu_hsv_8u_frame, cv2.COLOR_HSV2BGR)

        self.previous_gpu_frame = gpu_frame

        return gpu_bgr_frame