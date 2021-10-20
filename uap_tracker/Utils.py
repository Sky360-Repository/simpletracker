import cv2
import numpy as np

def get_cv_version():
    return (cv2.__version__).split('.')

def get_font_size():
    return 1

def get_writer(output_filename, width, height):
    print(f'source w,h:{(width, height)}')
    return cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"AVC1"), 30, (width, height))

def fisheye_mask(frame):
    shape = frame.shape[:2]
    # print(f'shape: {shape}')
    mask = np.zeros(shape, dtype="uint8")
    cv2.circle(mask, (int(shape[1] / 2), int(shape[0] / 2)), int(min(shape[0], shape[1]) * 0.46), 255, -1)
    return cv2.bitwise_and(frame, frame, mask=mask)

def createBackgroundSubtractorKNN():
    # defaults: samples:2, dist2Threshold:400.0, history: 500
    backSub = cv2.createBackgroundSubtractorKNN()
    samples = backSub.getkNNSamples()
    dist2Threshold = backSub.getDist2Threshold()
    history = backSub.getHistory()
    print(f'samples:{samples}, dist2Threshold:{dist2Threshold}, history, {history}')
    backSub.setHistory(1)  # large gets many detections
    backSub.setkNNSamples(2)  # 1 doesn't detect small object
    backSub.setDist2Threshold(5000)  # small gets many detections, large misses small movements
    return backSub

def kp_to_bbox(kp):
    (x, y) = kp.pt
    size = kp.size
    scale = 6
    return (int(x - scale * size / 2), int(y - scale * size / 2), int(scale * kp.size), int(scale * kp.size))

def bbox_overlap(bbox1, bbox2):
    #    bb1 : dict
    #        Keys: {'x1', 'x2', 'y1', 'y2'}
    #        The (x1, y1) position is at the top left corner,
    #        the (x2, y2) position is at the bottom right corner
    #    bb2 : dict
    #        Keys: {'x1', 'x2', 'y1', 'y2'}
    #        The (x, y) position is at the top left corner,
    #        the (x2, y2) position is at the bottom right corner

    bb1 = {}
    bb1['x1'] = bbox1[0]
    bb1['y1'] = bbox1[1]
    bb1['x2'] = bbox1[0] + bbox1[2]
    bb1['y2'] = bbox1[1] + bbox1[3]

    bb2 = {}
    bb2['x1'] = bbox2[0]
    bb2['y1'] = bbox2[1]
    bb2['x2'] = bbox2[0] + bbox2[2]
    bb2['y2'] = bbox2[1] + bbox2[3]

    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def is_bbox_being_tracked(live_trackers, bbox):
    # simple check to see if the new bbox intersects with an existing tracked bbox
    tracked = False
    for tracker in live_trackers:
        if tracker.does_bbx_overlap(bbox):
            tracked = True
            break

    return tracked

def detectSBD(frame, image_width):
    params = cv2.SimpleBlobDetector_Params()
    # print(f"original sbd params:{params}")

    params.minRepeatability = 2
    params.minDistBetweenBlobs = int(image_width * 0.05)  # 5% of the width of the image

    params.minThreshold = 3

    params.filterByArea = 1
    params.minArea = 5

    params.filterByColor = 0
    #    params.blobColor=255

    detector = cv2.SimpleBlobDetector_create(params)
    # params.write('params.json')
    # print("created detector")
    # blobframe=cv2.convertScaleAbs(frame)
    # print("blobframe")
    keypoints = detector.detect(frame)
    # print("ran detect")
    return keypoints
