import cv2
import numpy as np

def is_cv_version_supported():
    (major_ver, minor_ver, subminor_ver) = get_cv_version()
    if int(major_ver) >= 4 and int(minor_ver) >= 1:
        return True
    return False


def get_cv_version():
    return (cv2.__version__).split('.')


def resize_frame(resize, frame, w, h):
    if resize:
        #print(f"Applying Scaling to {w}, {h}")
        return scale_image_to(frame, w, h)
    return frame

def resize_frame_cuda(resize, gpu_frame, gpu_frame_w, gpu_frame_h, w, h):
    if resize:
        #print(f"Applying Scaling to {w}, {h}")
        return scale_image_to_cuda(gpu_frame, gpu_frame_w, gpu_frame_h, w, h)
    return gpu_frame


def get_writer(output_filename, width, height):
    print(f'source w,h:{(width, height)}')
    return cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"AVC1"), 30, (width, height))


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


def bbox1_contain_bbox2(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return (x2 > x1) and (y2 > y1) and (x2+w2 < x1+w1) and (y2+h2 < y1+h1)


def is_bbox_being_tracked(live_trackers, bbox):
    # MG: The bbox contained should computationally be faster than the overlap, so we use it first as a shortcut
    for tracker in live_trackers:
        if tracker.is_bbx_contained(bbox):
            return True
        else:
            if tracker.does_bbx_overlap(bbox):
                return True

    return False


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


def scale_image(frame, max_size_h_or_w):
    return scale_image_to(frame, max_size_h_or_w, max_size_h_or_w)

def scale_image_cuda(gpu_frame, gpu_frame_w, gpu_frame_h, max_size_h_or_w):
    return scale_image_to_cuda(gpu_frame, gpu_frame_w, gpu_frame_h, max_size_h_or_w, max_size_h_or_w)

def scale_image_to(frame, w, h):
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

def scale_image_to_cuda(gpu_frame, gpu_frame_w, gpu_frame_h, w, h):
    scale, scaled_height, scaled_width = calc_image_scale_cuda(gpu_frame_w, gpu_frame_h, w, h)
    if scale:
        gpu_frame = cv2.cuda.resize(gpu_frame, (scaled_width, scaled_height))
    return gpu_frame

# mask_pct - The percentage of the fisheye you want to mask


def apply_fisheye_mask(frame, mask_pct):
    mask_height = (100-mask_pct)/100.0
    mask_radius = mask_height/2.0
    shape = frame.shape[:2]
    height = shape[0]
    width = shape[1]
    new_width = int(mask_height * width)
    new_height = int(mask_height * height)
    mask = np.zeros(shape, dtype="uint8")
    cv2.circle(mask, (int(width / 2), int(height / 2)),
               int(min(height, width) * mask_radius), 255, -1)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    clipped_masked_frame = clip_at_center(
        masked_frame,
        (int(width/2), int(height/2)),
        width,
        height,
        new_width,
        new_height)
    return clipped_masked_frame


def apply_background_subtraction(frame_grey, background_subtractor):
    foreground_mask = background_subtractor.apply(frame_grey)
    return cv2.bitwise_and(frame_grey, frame_grey, mask=foreground_mask)

def apply_background_subtraction_cuda(gpu_frame_grey, background_subtractor, stream):
    gpu_foreground_mask = background_subtractor.apply(gpu_frame_grey, learningRate=0.05, stream=stream)
    return cv2.cuda.bitwise_and(gpu_frame_grey, gpu_frame_grey, mask=gpu_foreground_mask)

def add_bbox_to_image(bbox, frame, tracker_id, font_size, color):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, color, 2, 1)
    cv2.putText(frame, str(tracker_id),
                (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_TRIPLEX, font_size, color, 2)


def convert_to_grey(src, dst=None):
    weight = 1.0 / 3.0
    m = np.array([[weight, weight, weight]], np.float32)
    return cv2.transform(src, m, dst)


# Takes a frame and returns a smaller one
# (size divided by zoom level) centered on center
def zoom_and_clip(frame, center, zoom_level):
    height, width, _channels = frame.shape

    new_height = int(height/zoom_level)
    new_width = int(width/zoom_level)

    return clip_at_center(frame, center, width, height, new_width, new_height)


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


def combine_frames_2x2(top_left, top_right, bottom_left, bottom_right):
    im_h1 = cv2.hconcat([top_left, top_right])
    im_h2 = cv2.hconcat([bottom_left, bottom_right])
    return cv2.vconcat([im_h1, im_h2])


def stamp_original_frame(frame, font_size, font_color):
    cv2.putText(frame, 'Original Frame (Sky360)', (100, 200),
                cv2.FONT_HERSHEY_TRIPLEX, font_size, font_color, 2)


def stamp_output_frame(video_tracker, frame, font_size, font_color, fps):
    msg = f"Trackers: trackable:{sum(map(lambda x: x.is_tracking(), video_tracker.live_trackers))}, alive:{len(video_tracker.live_trackers)}, started:{video_tracker.total_trackers_started}, ended:{video_tracker.total_trackers_finished} (Sky360)"
    cv2.putText(frame, msg, (100, 200),
                cv2.FONT_HERSHEY_TRIPLEX, font_size, font_color, 2)
    cv2.putText(frame, f"FPS: {str(int(fps))} (Sky360)", (
        100, 300), cv2.FONT_HERSHEY_TRIPLEX, font_size, font_color, 2)


def display_frame(processed_frame, max_display_dim):
    # print(f"display_frame shape:{processed_frame.shape}, max:{max_display_dim}")
    # Display result, resize it to a standard size
    if processed_frame.shape[0] > max_display_dim or processed_frame.shape[1] > max_display_dim:
        # MG: scale the image to something that is of a reasonable viewing size
        frame_scaled = scale_image(
            processed_frame, max_display_dim)
        # print(f"{frame_scaled.shape}")
        cv2.imshow("Tracking", frame_scaled)
    else:
        cv2.imshow("Tracking", processed_frame)

def noise_reduction(noise_reduction, frame, blur_radius):
    noise_reduced_frame = frame
    if noise_reduction:
        #print(f"Applying noise reduction, blur radius:{blur_radius}")
        noise_reduced_frame = cv2.GaussianBlur(frame, (blur_radius, blur_radius), 0)
        # frame_grey = cv2.medianBlur(frame_grey, blur_radius)
    return noise_reduced_frame

def noise_reduction_cuda(noise_reduction, gpu_frame, blur_radius):
    gpu_noise_reduced_frame = gpu_frame
    if noise_reduction:
        #print(f"CUDA Applying noise reduction, blur radius:{blur_radius}")
        gpuFilter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (blur_radius, blur_radius), 0)
        gpu_noise_reduced_frame = cv2.cuda_Filter.apply(gpuFilter, gpu_frame)
    return gpu_noise_reduced_frame