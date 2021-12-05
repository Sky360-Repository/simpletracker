import cv2
import numpy as np
import uap_tracker.utils as utils


class DenseOpticalFlowCuda():

    def __init__(self, width, height):
        self.previous_gpu_frame = None
        self.width = width
        self.height = height

        # create hsv output for optical flow
        #hsv = np.zeros((self.height, self.width, 3), np.float32)
        # set saturation to 1
        #hsv[..., 1] = 1.0

        #self.gpu_hsv = cv2.cuda_GpuMat()
        #self.gpu_hsv.upload(hsv)

        # create gpu_hsv output for optical flow
        self.gpu_hsv_frame = cv2.cuda_GpuMat((self.height, self.width, 3), cv2.CV_32FC3)
        self.gpu_hsv_8u_frame = cv2.cuda_GpuMat((self.height, self.width, 3), cv2.CV_8UC3)

        # channels
        self.gpu_h_channel = cv2.cuda_GpuMat((self.height, self.width, 3), cv2.CV_32FC1)
        self.gpu_s_channel = cv2.cuda_GpuMat((self.height, self.width, 3), cv2.CV_32FC1)
        self.gpu_v_channel = cv2.cuda_GpuMat((self.height, self.width, 3), cv2.CV_32FC1)

        # set saturation to 1
        self.gpu_s_channel.upload(np.ones_like((self.height, self.width, 3), np.float32))

#    def process_frame(self, frame):
#        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        self.process_grey_frame(grey_frame)

    def _resize_cuda(self, gpu_frame):
        return cv2.cuda.resize(gpu_frame, (self.width, self.height))

    def process_grey_frame(self, gpu_frame):
        gpu_frame = self._resize_cuda(gpu_frame)

        gpu_bgr_frame = cv2.cuda_GpuMat()

        if self.previous_gpu_frame is None:
            # First time round, save frame and return blank image
            self.previous_gpu_frame = gpu_frame

            return gpu_bgr_frame.upload(np.zeros((self.height, self.width, 3), np.uint8))

        gpu_flow = cv2.cuda.calcOpticalFlowFarneback(
            self.previous_gpu_frame, gpu_frame, None, 0.5, 5, 15, 3, 5, 1.2, 0,
        )

        gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

        # convert from cartesian to polar coordinates to get magnitude and angle
        gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(gpu_flow_x, gpu_flow_y, angleInDegrees=True)

        # convert from cartesian to polar coordinates to get magnitude and angle
        #gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
        #    gpu_flow[..., 0], gpu_flow[..., 1], angleInDegrees=True,
        #)

        # set hue according to the angle of optical flow
        self.gpu_hsv[..., 0] = gpu_angle * ((1 / 360.0) * (180 / 255.0))

        # set value according to the normalized magnitude of optical flow
        self.gpu_hsv[..., 2] = gpu_magnitude
        #cv2.normalize(
        #    magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, -1,
        #)

        # multiply each pixel value to 255
        #hsv_8u = np.uint8(self.hsv * 255.0)
        self.gpu_hsv.convertTo(cv2.CV_8U, 255.0, self.gpu_hsv_8u, 0.0)

        # convert hsv to bgr
        gpu_bgr_frame = cv2.cuda.cvtColor(self.gpu_hsv_8u, cv2.COLOR_HSV2BGR)

        self.previous_gpu_frame = gpu_bgr_frame

        return gpu_bgr_frame
