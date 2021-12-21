import cv2
import numpy as np

class DenseOpticalFlow():

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def process_grey_frame(self, frame):
        pass

class DenseOpticalFlowCpu(DenseOpticalFlow):
    def __init__(self, width, height):
        super().__init__(width, height)

        self.previous_frame = None

        # create hsv output for optical flow
        self.hsv = np.zeros((self.height, self.width, 3), np.float32)

        # set saturation to 1
        self.hsv[..., 1] = 1.0

#    def process_frame(self, frame):
#        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        self.process_grey_frame(grey_frame)
    def _resize(self, frame):
        return cv2.resize(frame, (self.width, self.height))

    def process_grey_frame(self, frame):
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
