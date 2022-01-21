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


class BackgroundSubtractorFactory():

    @staticmethod
    def Select(enable_cuda, sensitivity):
        if enable_cuda:
            return BackgroundSubtractorFactory.create('MOG2_CUDA', sensitivity)

        return BackgroundSubtractorFactory.create('KNN', sensitivity)

    @staticmethod
    def create(type, sensitivity):
        background_subtractor = None
        if type == 'KNN':
            # defaults: samples:2, dist2Threshold:400.0, history: 500
            background_subtractor = cv2.createBackgroundSubtractorKNN()

            # samples = background_subtractor.getkNNSamples()
            # dist_2_threshold = background_subtractor.getDist2Threshold()
            # history = background_subtractor.getHistory()
            # print(f'samples:{samples}, dist2Threshold:{dist_2_threshold}, history, {history}')

            #background_subtractor.setHistory(1)  # large gets many detections

            # setkNNSamples() --> Sets the k in the kNN. How many nearest neighbours need to match
            #  -  1 doesn't detect small object
            # setDist2Threshold() --> Sets the threshold on the squared distance.
            #  -  small gets many detections, large misses small movements

            if sensitivity == 1:  # Detects small, medium and large objects
                background_subtractor.setkNNSamples(2)
                background_subtractor.setDist2Threshold(500)
            elif sensitivity == 2:  # Detects medium and large objects
                background_subtractor.setkNNSamples(2)
                background_subtractor.setDist2Threshold(5000)
            elif sensitivity == 3:  # Detects large objects
                background_subtractor.setkNNSamples(1)
                background_subtractor.setDist2Threshold(5000)
            else:
                raise Exception(f"Unknown sensitivity option ({sensitivity}). 1, 2 and 3 is supported not {sensitivity}.")

        if type == 'MOG2':
            background_subtractor = cv2.createBackgroundSubtractorMOG2()
            #background_subtractor.setHistory(1)  # large gets many detections
            if sensitivity == 1:  # Detects small, medium and large objects
                background_subtractor.setVarThreshold(15)
            elif sensitivity == 2:  # Detects medium and large objects
                background_subtractor.setVarThreshold(25)
            elif sensitivity == 3:  # Detects large objects
                background_subtractor.setVarThreshold(50)
            else:
                raise Exception(f"Unknown sensitivity option ({sensitivity}). 1, 2 and 3 is supported not {sensitivity}.")

        if type == 'MOG2_CUDA':
            if sensitivity == 1:  # Detects small, medium and large objects
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(varThreshold=1500) #, detectShadows=True)
            elif sensitivity == 2:  # Detects medium and large objects
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(varThreshold=2000) #, detectShadows=True)
            elif sensitivity == 3:  # Detects large objects
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(varThreshold=2000) #, detectShadows=True)
            else:
                raise Exception(f"Unknown sensitivity option ({sensitivity}). 1, 2 and 3 is supported not {sensitivity}.")

        if background_subtractor is None:
            raise Exception(f"Unknown background subtractor type ({type}).")
        else:
            background_subtractor.setHistory(1)  # large gets many detections

        return background_subtractor
