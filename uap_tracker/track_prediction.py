# Original work Copyright (c) 2022 Sky360
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

import cv2
import numpy as np

#https://pysource.com/2021/11/02/kalman-filter-predict-the-trajectory-of-an-object/

####################################################################################
# This class represents an attempt to predict a futur epoint using a Kalman Filter #
####################################################################################
class TrackPrediction():

    def __init__(self, id, bbox):

        self.id = id
        x, y, w, h = bbox

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
        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

    # update function, we notify where we currently are and ask for a future prediction point
    def update(self, bbox):

        x, y, w, h = bbox
        center = np.array([x+w/2, y+h/2], np.float32)
        self.kalman.correct(center)
        predicted = self.kalman.predict()

        x, y = int(predicted[0]), int(predicted[1])
        return x, y
