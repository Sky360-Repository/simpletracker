import cv2

#
# Factory that creates background subtractors
#
class BackgroundSubtractorFactory():

    @staticmethod
    def create(type, sensitivity):
        if type == 'KNN':
            # defaults: samples:2, dist2Threshold:400.0, history: 500
            background_subtractor = cv2.createBackgroundSubtractorKNN()

            # samples = background_subtractor.getkNNSamples()
            # dist_2_threshold = background_subtractor.getDist2Threshold()
            # history = background_subtractor.getHistory()
            # print(f'samples:{samples}, dist2Threshold:{dist_2_threshold}, history, {history}')

            background_subtractor.setHistory(1)  # large gets many detections

            if sensitivity == 1:  # Detects small, medium and large objects
                background_subtractor.setkNNSamples(2)  # 1 doesn't detect small object
                background_subtractor.setDist2Threshold(500)  # small gets many detections, large misses small movements
            elif sensitivity == 2:  # Detects medium and large objects
                background_subtractor.setkNNSamples(2)  # 1 doesn't detect small object
                background_subtractor.setDist2Threshold(5000)  # small gets many detections, large misses small movements
            elif sensitivity == 3:  # Detects large objects
                background_subtractor.setkNNSamples(1)  # 1 doesn't detect small object
                background_subtractor.setDist2Threshold(5000)  # small gets many detections, large misses small movements
            else:
                raise Exception(f"Unknown sensitivity option ({sensitivity}). 1, 2 and 3 is supported not {sensitivity}.")
        else:
            raise Exception('Only the KNN Background Subtractor is currently supported')

        return background_subtractor