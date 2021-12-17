import cv2

#
# Factory that creates background subtractors
#
class BackgroundSubtractorFactory():

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
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(varThreshold=500) #, detectShadows=True)
            elif sensitivity == 2:  # Detects medium and large objects
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(varThreshold=1000) #, detectShadows=True)
            elif sensitivity == 3:  # Detects large objects
                background_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(varThreshold=1500) #, detectShadows=True)
            else:
                raise Exception(f"Unknown sensitivity option ({sensitivity}). 1, 2 and 3 is supported not {sensitivity}.")

        if background_subtractor is None:
            raise Exception(f"Unknown background subtractor type ({type}).")
        else:
            background_subtractor.setHistory(1)  # large gets many detections

        return background_subtractor
