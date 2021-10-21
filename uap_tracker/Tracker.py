import cv2
import Utils as u

#
# Tracks a single object
#
class Tracker():

    def __init__(self, id, tracker_type):
        self.cv2_tracker = Tracker.create_cv2_tracker(tracker_type)
        self.id = id
        self.bboxes = []

    @staticmethod
    def create_cv2_tracker(tracker_type):
        tracker = None
        (major_ver, minor_ver, subminor_ver) = u.get_cv_version()
        if int(minor_ver) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        else:
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

    def update_bbox(self, bbox):
        return self.bboxes.append(bbox)

    def get_bbox(self):
        return self.bboxes[-1]

    def add_bbox_to_image(self, image, font_size, color):
        bbox = self.get_bbox()
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(image, p1, p2, color, 2, 1)
        cv2.putText(image, str(self.id), (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)

    def does_bbx_overlap(self, bbox):
        overlap = u.bbox_overlap(self.get_bbox(), bbox)
        # print(f'checking tracking overlap {overlap} for {self.id}')
        return overlap > 0
