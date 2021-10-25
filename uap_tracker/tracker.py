import cv2
import numpy as np
import uap_tracker.utils as utils
from uap_tracker.tracker_factory import TrackerFactory

#
# Tracks a single object
#
class Tracker():

    def __init__(self, id, tracker_type, frame, frame_hsv, bbox, font_size, font_color):

        self.id = id
        self.cv2_tracker = TrackerFactory.create(tracker_type)
        self.cv2_tracker.init(frame, bbox)
        self.bboxes = [bbox]
        self.font_size = font_size
        self.font_color = font_color

        self.track_window = bbox

        self.enable_kalman = False

        if self.enable_kalman:
            # MG TODO: Move this kalman stuff out into a listener
            self.term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

            # Initialize the histogram.
            x, y, w, h = bbox
            roi = frame_hsv[y:y+h, x:x+w]
            roi_hist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            # Initialize the Kalman filter.
            self.kalman = cv2.KalmanFilter(4, 2)
            self.kalman.measurementMatrix = np.array(
                [[1, 0, 0, 0],
                 [0, 1, 0, 0]], np.float32)
            self.kalman.transitionMatrix = np.array(
                [[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], np.float32)
            self.kalman.processNoiseCov = np.array(
                [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], np.float32) * 0.03
            cx = x+w/2
            cy = y+h/2
            self.kalman.statePre = np.array(
                [[cx], [cy], [0], [0]], np.float32)
            self.kalman.statePost = np.array(
                [[cx], [cy], [0], [0]], np.float32)

    def get_bbox(self):
        return self.bboxes[-1]

    def update(self, frame, frame_hsv):
        ok, bbox = self.cv2_tracker.update(frame)
        if ok:
            self.bboxes.append(bbox)
            self.track_window = bbox

            utils.add_bbox_to_image(bbox, frame, self.id, self.font_size, self.font_color)

            if self.enable_kalman:
                # MG: The meanShift option of tracking does not work very well for us, but I am keeping the kalman stuff for now.
                # back_proj = cv2.calcBackProject([frame_hsv], [0], self.roi_hist, [0, 180], 1)
                # ret, self.track_window = cv2.meanShift(back_proj, self.track_window, self.term_crit)
                x, y, w, h = self.track_window
                center = np.array([x+w/2, y+h/2], np.float32)

                prediction = self.kalman.predict()
                estimate = self.kalman.correct(center)
                center_offset = estimate[:,0][:2] - center
                self.track_window = (x + int(center_offset[0]), y + int(center_offset[1]), w, h)

                # Draw the predicted center position as a blue circle.
                cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (255, 0, 0), -1)

                # x, y, w, h = self.track_window

                # Draw the corrected tracking window as a cyan rectangle.
                # cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

                # Draw the ID above the rectangle in blue text.
                # cv2.putText(frame, f'MS ID: {self.id}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

        return ok, bbox

    def does_bbx_overlap(self, bbox):
        overlap = utils.bbox_overlap(self.bboxes[-1], bbox)
        # print(f'checking tracking overlap {overlap} for {self.id}')
        return overlap > 0
