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

###########################################################################################################
# This code file contains a selection of useful utility functions that are used throughout simple tracker #
###########################################################################################################

# Utility function to determine if the installed version of open cv is supported
# We support v4.1 and above
def is_cv_version_supported():
    (major_ver, minor_ver, subminor_ver) = get_cv_version()
    if int(major_ver) >= 4 and int(minor_ver) >= 1:
        return True
    return False

# Utility function to get the installed version of open cv
def get_cv_version():
    return (cv2.__version__).split('.')

# Utility function to determine if cuda is supported
def is_cuda_supported(): # 1 == using cuda, 0 = not using cuda
    enabled = False
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            enabled = True
    except:
        pass
    return enabled

# Utility function to get the video writer in a standardised way
def get_writer(output_filename, width, height):
    print(f'source w,h:{(width, height)}')
    return cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"AVC1"), 30, (width, height))

# Utility function to convert jey points in to a bounding box
# The bounding box is used for track validation (if enabled) and will be displayed by the visualiser
# as it tracks a point of interest (blob) on the frame
def kp_to_bbox(kp):
    (x, y) = kp.pt
    size = kp.size
    scale = 6
    #print(f'kp_to_bbox x, y:{(x, y)}, size:{size}, scale:{scale}, new size:{scale * size}')
    return (int(x - scale * size / 2), int(y - scale * size / 2), int(scale * size), int(scale * size))

# Utility function to determine if 2 bounding boxes overlap each other. In order to make tracking more efficient
# we try not to track sections of the same point of interest (blob)
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

# Utility function to determine if a bounding box 1 contains bounding box 2. In order to make tracking more efficient
# we try not to track sections of the same point of interest (blob)
def bbox1_contain_bbox2(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return (x2 > x1) and (y2 > y1) and (x2+w2 < x1+w1) and (y2+h2 < y1+h1)

# Utility function to determine if a bounding box is already being tracked by checkling if its overlapped or already contained
def is_bbox_being_tracked(live_trackers, bbox):
    # Mike: The bbox contained should computationally be faster than the overlap, so we use it first as a shortcut
    for tracker in live_trackers:
        if tracker.is_bbx_contained(bbox):
            return True
        else:
            if tracker.does_bbx_overlap(bbox):
                return True

    return False

# Utility function to detect blobs in a background subtracted frame
def perform_blob_detection(frame, sensitivity):
    params = cv2.SimpleBlobDetector_Params()
    # print(f"original sbd params:{params}")

    params.minRepeatability = 2
    # 5% of the width of the image
    params.minDistBetweenBlobs = int(frame.shape[1] * 0.05)
    params.minThreshold = 3
    params.filterByArea = 1
    params.filterByColor = 0
    # params.blobColor=255

    if sensitivity == 1:  # Detects small, medium and large objects
        params.minArea = 3
    elif sensitivity == 2:  # Detects medium and large objects
        params.minArea = 5
    elif sensitivity == 3:  # Detects large objects
        params.minArea = 25
    else:
        raise Exception(
            f"Unknown sensitivity option ({sensitivity}). 1, 2 and 3 is supported not {sensitivity}.")

    detector = cv2.SimpleBlobDetector_create(params)
    # params.write('params.json')
    # print("created detector")
    # blobframe=cv2.convertScaleAbs(frame)
    # print("blobframe")
    keypoints = detector.detect(frame)
    # print("ran detect")
    return keypoints

# Check to see if an image needs to be scaled down
def calc_image_scale(frame_w, frame_h, to_w, to_h):
    if frame_h > to_h or frame_w > to_w:
        # calculate the width and height percent of original size
        width = int((to_w / frame_w) * 100)
        height = int((to_h / frame_h) * 100)
        # pick the largest of the two
        scale_percent = max(width, height)
        # calc the scaled width and height
        scaled_width = int(frame_w * scale_percent / 100)
        scaled_height = int(frame_h * scale_percent / 100)
        return (True, scaled_width, scaled_height)
    else:
        return (False,  frame_w, frame_h)

# Utility function to standardise the drawing of a bounding box (rectangle) onto a frame
def add_bbox_to_image(bbox, frame, tracker_id, font_size, color, settings):
    x1, y1, w, h = get_sized_bbox(bbox, settings)
    p1 = (int(x1), int(y1))
    p2 = (int(x1 + w), int(y1 + h))
    cv2.rectangle(frame, p1, p2, color, 2, 1)
    cv2.putText(frame, str(tracker_id), (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)

# Utility function to standardise the drawing of the track center point onto a frame
def add_track_points_to_image(tracker, frame):
    bbox = get_sized_bbox_from_tracker(tracker)
    center_points = tracker.center_points
    for center_point in center_points:
        if not is_point_contained_in_bbox(bbox, center_point[0]):
            cv2.circle(frame, center_point[0], radius=1, color=center_point[1], thickness=2)

# Utility function to standardise the drawing of the track line onto a frame
def add_track_line_to_image(tracker, frame):
    bbox = get_sized_bbox_from_tracker(tracker)
    center_points = tracker.center_points
    previous_point = None
    for center_point in center_points:
        if not previous_point is None:
            if not is_point_contained_in_bbox(bbox, center_point[0]):
                cv2.line(frame, previous_point[0], center_point[0], previous_point[1], thickness=2)
        previous_point = center_point

# Utility function to standardise the drawing of bbox center point onto a frame
def add_center_point_to_image(tracker, frame):
    predicted_center_point = tracker.get_center()
    cv2.circle(frame, predicted_center_point, radius=1, color=(0, 0, 255), thickness=2)

# Utility function to standardise the drawing of prediction point onto a frame
def add_predicted_point_to_image(tracker, frame):
    predicted_center_point = tracker.predictor_center_points[-1]
    cv2.circle(frame, predicted_center_point, radius=1, color=(255, 0, 0), thickness=2)

# Utility function to deletrmine if a point overlaps a bouding box
def is_point_contained_in_bbox(bbox, point):
    x, y, w, h = bbox
    x0, y0 = point
    return x <= x0 < x + w and y <= y0 < y + h

# Utility function to get the sized bbox from tracker for display
def get_sized_bbox_from_tracker(tracker):
    return get_sized_bbox(tracker.get_bbox(), tracker.settings)

# Utility function to get the sized bbox for display
def get_sized_bbox(bbox, settings):
    return_bbox = bbox
    if settings['bbox_fixed_size']:
        size = settings['bbox_size']
        x1, y1, w, h = bbox
        x1 = int(x1+(w/2)) - int(size/2)
        y1 = int(y1+(h/2)) - int(size/2)
        return_bbox = (x1, y1, size, size)

    return return_bbox

# Utility function to take a frame and return a smaller one
# (size divided by zoom level) centered on center
def zoom_and_clip(frame, center, zoom_level):
    height, width, _channels = frame.shape

    new_height = int(height/zoom_level)
    new_width = int(width/zoom_level)

    return clip_at_center(frame, center, width, height, new_width, new_height)

# Utility function to clip out the center part of a frame. This is mainly used by the fish-eye mask
# to remove masked ("black") parts of the frame
def clip_at_center(frame, center, width, height, new_width, new_height):
    x, y = center
    half_width = int(new_width/2)
    half_height = int(new_height/2)

    left = max(0, x-half_width)
    right = min(x+half_width, width)
    right = max(new_width, right)

    top = max(0, y-half_height)
    bottom = min(y+half_height, height)
    bottom = max(new_height, bottom)

    return frame[top:bottom, left:right]

# Utility function to combine 4 frames into a single frame, mainly used by a visualiser
def combine_frames_2x2(top_left, top_right, bottom_left, bottom_right):
    im_h1 = cv2.hconcat([top_left, top_right])
    im_h2 = cv2.hconcat([bottom_left, bottom_right])
    return cv2.vconcat([im_h1, im_h2])

# Utility function to stamp the original frame with text stating its the original frame
def stamp_original_frame(frame, font_size, font_color, font_thickness):
    cv2.putText(frame, 'Original Frame (Sky360)', (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

# Utility function to stamp the output frame with text displaying the current state of the trackers
def stamp_output_frame(video_tracker, frame, font_size, font_color, fps, font_thickness):
    msg = f"Trackers: trackable:{sum(map(lambda x: x.is_tracking(), video_tracker.live_trackers))}, alive:{len(video_tracker.live_trackers)}, started:{video_tracker.total_trackers_started}, ended:{video_tracker.total_trackers_finished} (Sky360)"
    cv2.putText(frame, msg, (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(frame, f"FPS: {str(int(fps))} (Sky360)", (
        25, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

# Utility function to standardise the display of a frame
def display_frame(processed_frame, max_display_dim):
    # print(f"display_frame shape:{processed_frame.shape}, max:{max_display_dim}")
    # Display result, resize it to a standard size
    if processed_frame.shape[0] > max_display_dim or processed_frame.shape[1] > max_display_dim:
        # Mike: scale the image to something that is of a reasonable viewing size
        frame_scaled = _scale_image_for_display(
            processed_frame, max_display_dim, max_display_dim)
        # print(f"{frame_scaled.shape}")
        cv2.imshow("Tracking", frame_scaled)
    else:
        cv2.imshow("Tracking", processed_frame)

# Utility function to standardise the scaling of an image for display
# NOTE: Mike: This will likely be removed in future as it does not take platform (cpu or gpu) into consideration
def _scale_image_for_display(frame, w, h):
    if frame.shape[0] > h or frame.shape[1] > w:
        # calculate the width and height percent of original size
        width = int((w / frame.shape[1]) * 100)
        height = int((h / frame.shape[0]) * 100)
        # pick the largest of the two
        scale_percent = max(width, height)
        # calc the scaled width and height
        scaled_width = int(frame.shape[1] * scale_percent / 100)
        scaled_height = int(frame.shape[0] * scale_percent / 100)
        return cv2.resize(frame, (scaled_width, scaled_height))
    else:
        return frame
