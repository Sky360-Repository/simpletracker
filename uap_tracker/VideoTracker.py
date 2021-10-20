import cv2
import sys
import numpy as np
import Utils as u
import Tracker as t

#
# Tracks multiple objects in a video
#
class VideoTracker():

    def __init__(self, video):
        # print(f'VideoTracker called {video}')
        self.video = video
        self.total_trackers_finished = 0
        self.total_trackers_started = 0
        self.live_trackers = []
        self.listeners = []

    def listen(self, listener):
        self.listeners.append(listener)

    def create_trackers_from_keypoints(self, tracker_type, keypoints, frame):
        trackers = []
        for kp in keypoints:
            bbox = u.kp_to_bbox(kp)
            print(bbox)

            # Initialize tracker with first frame and bounding box
            if not u.is_bbox_being_tracked(self.live_trackers, bbox):
                self.create_and_add_tracker(tracker_type, frame, bbox)

        print(self.live_trackers)

    def create_and_add_tracker(self, tracker_type, frame, bbox):
        if not bbox:
            raise Exception("null bbox")

        self.total_trackers_started += 1

        tracker = t.Tracker(self.total_trackers_started, tracker_type)
        tracker.cv2_tracker.init(frame, bbox)
        tracker.update_bbox(bbox)
        self.live_trackers.append(tracker)

    def update_trackers(self, tracker_type, keypoints, masked):

        # cache kp -> bbox mapping for removing failed trackers
        kp_bbox_map = {}
        for kp in keypoints:
            bbox = u.kp_to_bbox(kp)
            kp_bbox_map[kp] = bbox

        failed_trackers = []
        for idx, tracker in enumerate(self.live_trackers):
            # Update tracker
            ok, bbox = tracker.cv2_tracker.update(masked)

            if ok:
                # Tracking success
                # tracker_success(tracker, bbox, masked)
                tracker.update_bbox(bbox)
            else:
                # Tracking failure
                failed_trackers.append(tracker)

            # Try to match the new detections with this tracker
            for idx, kp in enumerate(keypoints):
                if kp in kp_bbox_map:
                    overlap = u.bbox_overlap(bbox, kp_bbox_map[kp])
                    # print(f'Overlap: {overlap}; bbox:{bbox}, new_bbox:{new_bbox}')
                    if overlap > 0.2:
                        del (kp_bbox_map[kp])

        # remove failed trackers from live tracking
        for tracker in failed_trackers:
            self.live_trackers.remove(tracker)
            self.total_trackers_finished += 1

        # Add new detections to live tracker
        max_trackers = 10
        for kp, new_bbox in kp_bbox_map.items():
            # Hit max trackers?
            if len(self.live_trackers) < max_trackers:
                if not u.is_bbox_being_tracked(self.live_trackers, new_bbox):
                    self.create_and_add_tracker(tracker_type, masked, new_bbox)

    def detect_and_track(self, trackers_updated_callback=None, record=True, demo_mode=False, two_by_two=False):

        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'DASIAMRPN']
        tracker_type = tracker_types[7]

        # Open output video
        source_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = u.get_writer("outputvideo.mp4", source_width, source_height)

        font_size = source_height / 1000.0

        # Read first frame.
        ok, frame = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        backSub = u.createBackgroundSubtractorKNN()
        output_image = u.fisheye_mask(frame)

        def bg_subtract(frame, background_subtractor):
            a_masked = u.fisheye_mask(frame)
            a_fgMask = background_subtractor.apply(a_masked)
            return (a_masked, cv2.bitwise_and(a_masked, a_masked, mask=a_fgMask))

        output_image, bgMasked = bg_subtract(frame, backSub)

        for i in range(5):
            ok, frame = self.video.read()
            output_image, bgMasked = bg_subtract(frame, backSub)

        keypoints = u.detectSBD(bgMasked, source_width)

        print(keypoints)
        im_with_keypoints = cv2.drawKeypoints(output_image, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        msg = f"Detected {len(keypoints)} keypoints: "
        # print(msg)
        cv2.putText(im_with_keypoints, msg, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50, 170, 50), 2)
        # cv2.imshow('mask', im_with_keypoints)
        # cv2.waitKey()

        if demo_mode:
            for i in range(150):
                cv2.putText(frame, "First frame, look for BLOB's: ", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                            (50, 170, 50), 2)
                writer.write(frame)

            for i in range(150):
                cv2.putText(im_with_keypoints, "Identified here with in red: ", (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, (50, 170, 50), 2)
                writer.write(im_with_keypoints)

        # Create Trackers
        self.create_trackers_from_keypoints(tracker_type, keypoints, output_image)

        frame_count = 0
        while True:
            # Read a new frame
            ok, frame = self.video.read()
            if not ok:
                break

            # Start timer
            timer = cv2.getTickCount()

            output_image, bgMasked = bg_subtract(frame, backSub)

            # Detect new objects of interest to pass to tracker
            keypoints = u.detectSBD(bgMasked, source_width)

            im_with_keypoints = cv2.drawKeypoints(bgMasked, keypoints, np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            msg = f"Detected {len(keypoints)} keypoints: "
            # print(msg)
            cv2.putText(im_with_keypoints, msg, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50, 170, 50), 2)

            self.update_trackers(tracker_type, keypoints, output_image)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            for listener in self.listeners:
                listener.trackers_updated_callback(output_image, self.live_trackers, fps)

            for tracker in self.live_trackers:
                tracker.add_bbox_to_image(output_image, (0, 255, 0))

            msg = f"Trackers: started:{self.total_trackers_started}, ended:{self.total_trackers_finished}, alive:{len(self.live_trackers)}"
            print(msg)
            cv2.putText(output_image, msg, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50, 170, 50), 2)
            cv2.putText(output_image, "FPS : " + str(int(fps)), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                        (50, 170, 50), 2)

            if two_by_two:
                im_h1 = cv2.hconcat([frame, output_image])
                im_h2 = cv2.hconcat([bgMasked, im_with_keypoints])

                final_image = cv2.vconcat([im_h1, im_h2])
            else:
                final_image = output_image

            # final_image=im_v2

            # Display result, resize it to a standard size
            if source_width > 1280 or source_height > 1280:

                #  scale to a reasonable viewing size
                w = int((1280 / source_width) * 100)
                h = int((1280 / source_height) * 100)
                scale_percent = min(w, h)  # percent of original size
                width = int(source_width * scale_percent / 100)
                height = int(source_height * scale_percent / 100)

                final_resized_image = cv2.resize(final_image, (width, height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Tracking", final_resized_image)
            else:
                cv2.imshow("Tracking", final_image)

            if record or demo_mode:
                writer.write(final_image)

            # keyboard = cv2.waitKey()

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

            frame_count += 1

        writer.release()

        for listener in self.listeners:
            listener.finish(self.total_trackers_started, self.total_trackers_finished)