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

####################################################################################################################
# This class provides a factory implimentation for selecting which background subtraction algorithm should be used #
#   We use KNN by default, unfortunately KNN has no CUDA implementation so for CUDA we deafult to MOG2             #
####################################################################################################################
class BackgroundSubtractorFactory():

    # Static factory select method to determine what background subtraction algorithm to use
    # We use KNN by default, unfortunately KNN has no CUDA implementation so for CUDA we deafult to MOG2
    @staticmethod
    def Select(settings):
        enable_cuda = settings['enable_cuda']
        if enable_cuda:
            return BackgroundSubtractorFactory.create('MOG2_CUDA', settings)

        return BackgroundSubtractorFactory.create('KNN', settings)

    # Static create method, used to instantiate the selected background subtraction algorithm along with
    # whatever parameters that have been configured
    #
    # Mike: I am not sure about the parameters going into the background subtractors, this is all still work in process and quite experimental
    @staticmethod
    def create(type, settings):
        sensitivity = settings['detection_sensitivity']
        background_subtractor = None
        if type == 'KNN':
            background_subtractor = cv2.createBackgroundSubtractorKNN()

            #background_subtractor.setHistory(1)  # large gets many detections

            # setkNNSamples() --> Sets the k in the kNN. How many nearest neighbours need to match
            #  -  1 doesn't detect small object
            # setDist2Threshold() --> Sets the threshold on the squared distance.
            #  -  small gets many detections, large misses small movements

            if sensitivity == 1:  # Detects small, medium and large objects
                background_subtractor.setkNNSamples(2)
                background_subtractor.setDist2Threshold(500)
                background_subtractor.setHistory(1)  # large gets many detections
            elif sensitivity == 2:  # Detects medium and large objects
                background_subtractor.setkNNSamples(2)
                background_subtractor.setDist2Threshold(5000)
                background_subtractor.setHistory(1)  # large gets many detections
            elif sensitivity == 3:  # Detects large objects
                background_subtractor.setkNNSamples(1)
                background_subtractor.setDist2Threshold(5000)
                background_subtractor.setHistory(1)  # large gets many detections
            else:
                raise Exception(f"Unknown sensitivity option ({sensitivity}). 1, 2 and 3 is supported not {sensitivity}.")

        if type == 'MOG':
            #background_subtractor = cv2.createBackgroundSubtractorMOG(history=200, varThreshold=16, detectShadows=False)
            #background_subtractor.setHistory(1)  # large gets many detections
            if sensitivity == 1:  # Detects small, medium and large objects
                background_subtractor = cv2.createBackgroundSubtractorMOG(history=1, nmixtures=15, backgroundRatio=0.7, noiseSigma=0)
            elif sensitivity == 2:  # Detects medium and large objects
                background_subtractor = cv2.createBackgroundSubtractorMOG(history=1, nmixtures=25, backgroundRatio=0.7, noiseSigma=0)
            elif sensitivity == 3:  # Detects large objects
                background_subtractor = cv2.createBackgroundSubtractorMOG(history=1, nmixtures=50, backgroundRatio=0.7, noiseSigma=0)
            else:
                raise Exception(f"Unknown sensitivity option ({sensitivity}). 1, 2 and 3 is supported not {sensitivity}.")

        if type == 'MOG_CUDA':
            if sensitivity == 1:  # Detects small, medium and large objects
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG(history=1, nmixtures=15, backgroundRatio=0.7, noiseSigma=0)
            elif sensitivity == 2:  # Detects medium and large objects
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG(history=1, nmixtures=25, backgroundRatio=0.7, noiseSigma=0)
            elif sensitivity == 3:  # Detects large objects
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG(history=1, nmixtures=50, backgroundRatio=0.7, noiseSigma=0)
            else:
                raise Exception(f"Unknown sensitivity option ({sensitivity}). 1, 2 and 3 is supported not {sensitivity}.")

        if type == 'MOG2':
            #background_subtractor = cv2.createBackgroundSubtractorMOG2()
            #background_subtractor.setHistory(1)  # large gets many detections
            if sensitivity == 1:  # Detects small, medium and large objects
                background_subtractor = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=15, detectShadows=False)
            elif sensitivity == 2:  # Detects medium and large objects
                background_subtractor = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=25, detectShadows=False)
            elif sensitivity == 3:  # Detects large objects
                background_subtractor = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=50, detectShadows=False)
            else:
                raise Exception(f"Unknown sensitivity option ({sensitivity}). 1, 2 and 3 is supported not {sensitivity}.")

        if type == 'MOG2_CUDA':
            if sensitivity == 1:  # Detects small, medium and large objects
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=1, varThreshold=1500, detectShadows=False)
            elif sensitivity == 2:  # Detects medium and large objects
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=1, varThreshold=2000, detectShadows=False)
            elif sensitivity == 3:  # Detects large objects
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=1, varThreshold=2000, detectShadows=False)
            else:
                raise Exception(f"Unknown sensitivity option ({sensitivity}). 1, 2 and 3 is supported not {sensitivity}.")

        if background_subtractor is None:
            raise Exception(f"Unknown background subtractor type ({type}).")

        return background_subtractor
