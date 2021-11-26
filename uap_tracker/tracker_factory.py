import cv2
import uap_tracker.utils as utils

#
# Factory that creates trackers
#
class TrackerFactory():

    @staticmethod
    def create(tracker_type):
        tracker = None
        (major_ver, minor_ver, subminor_ver) = utils.get_cv_version()

        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            if int(minor_ver) == 1:
                tracker = cv2.TrackerCSRT_create()
            elif  int(minor_ver) > 1:
                param_handler = cv2.TrackerCSRT_Params()
                param_handler.use_gray = True
                # print(f"psr_threshold: {param_handler.psr_threshold}")
                param_handler.psr_threshold = 0.06
                # fs = cv2.FileStorage("csrt_defaults.json", cv2.FileStorage_WRITE)
                # param_handler.write(fs)
                # fs.release()
                # param_handler.use_gray=True
                tracker = cv2.TrackerCSRT_create(param_handler)
        if tracker_type == 'DASIAMRPN':
            tracker = cv2.TrackerDaSiamRPN_create()

        return tracker
