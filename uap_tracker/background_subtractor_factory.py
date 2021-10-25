import cv2

#
# Factory that creates background subtractors
#
class BackgroundSubtractorFactory():

    @staticmethod
    def create(type, sensitivity=2):
        if type == 'KNN':
            # defaults: samples:2, dist2Threshold:400.0, history: 500
            background_subtractor = cv2.createBackgroundSubtractorKNN()
            # samples = background_subtractor.getkNNSamples()
            # dist_2_threshold = background_subtractor.getDist2Threshold()
            # history = background_subtractor.getHistory()
            # print(f'samples:{samples}, dist2Threshold:{dist_2_threshold}, history, {history}')
            # MG TODO If clause here to handle the sensitivity parameter e.g. how aggressive do we want the subtraction to be
            background_subtractor.setHistory(1)  # large gets many detections
            background_subtractor.setkNNSamples(2)  # 1 doesn't detect small object
            background_subtractor.setDist2Threshold(5000)  # small gets many detections, large misses small movements
        else:
            raise Exception('Only the KNN Background Subtractor is currently supported')

        return background_subtractor