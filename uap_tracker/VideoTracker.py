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

    @property
    def is_tracking(self):
        return len(self.live_trackers) > 0

    def create_trackers_from_keypoints(self, tracker_type, keypoints, frame):
        for kp in keypoints:
            bbox = u.kp_to_bbox(kp)
            # print(bbox)

            # Initialize tracker with first frame and bounding box
            if not u.is_bbox_being_tracked(self.live_trackers, bbox):
                self.create_and_add_tracker(tracker_type, frame, bbox)

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
        max_display_dim = u.get_image_max_display_size_h_or_w()

        # Read first frame.
        ok, frame = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        backSub = u.createBackgroundSubtractorKNN()
        # output_image = u.fisheye_mask(frame_gray)  # MG: I have commented this out as it does nothing

        def bg_subtract(frame_gray, background_subtractor):
            a_masked = u.fisheye_mask(frame_gray)
            a_fgMask = background_subtractor.apply(a_masked)
            return a_masked, cv2.bitwise_and(a_masked, a_masked, mask=a_fgMask)

        output_image, bgMasked = bg_subtract(frame_gray, backSub)

        for i in range(5):
            ok, frame = self.video.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output_image, bgMasked = bg_subtract(frame_gray, backSub)

        keypoints = u.detectSBD(bgMasked, source_width)
        # print(keypoints)

        if demo_mode:  # MG: Moved this into the if block so that it only runs when needed. Paul --> DELETE THIS COMMENT IF YOU OKAY WITH THIS CHANGE

            im_with_keypoints = cv2.drawKeypoints(output_image, keypoints, np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            msg = f"Detected {len(keypoints)} keypoints: "
            # print(msg)
            cv2.putText(im_with_keypoints, msg, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50, 170, 50), 2)
            # cv2.imshow('mask', im_with_keypoints)
            # cv2.waitKey()

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

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # MG: This needs to be done on an 8 bit gray scale image, the colour image is causing a detection cluster
            output_image, bgMasked = bg_subtract(frame_gray, backSub)
            output_image = frame  # MG I have set the output image to be that of the original frame so that we preserve colour i.e. 24 bit image

            # Detect new objects of interest to pass to tracker
            keypoints = u.detectSBD(bgMasked, source_width)

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
            cv2.putText(output_image, "FPS : " + str(int(fps)), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50, 170, 50), 2)

            if two_by_two:
                # MG: trying to speed up the loop so moved this code here as im_with_keypoints appears to only be used here within the loop
                im_with_keypoints = cv2.drawKeypoints(bgMasked, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                msg = f"Detected {len(keypoints)} keypoints: "
                # print(msg)
                cv2.putText(im_with_keypoints, msg, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50, 170, 50), 2)

                im_h1 = cv2.hconcat([frame, output_image])
                im_h2 = cv2.hconcat([bgMasked, im_with_keypoints])

                final_image = cv2.vconcat([im_h1, im_h2])
            else:
                final_image = output_image

            # Display result, resize it to a standard size

            if final_image.shape[0] > max_display_dim or final_image.shape[1] > max_display_dim:
                #  MG: scale the image to something that is of a reasonable viewing size but write the original to file
                scaled_image = u.scaleImage(final_image, max_display_dim)
                cv2.imshow("Tracking", scaled_image)
            else:
                cv2.imshow("Tracking", final_image)

            if record or demo_mode:
                writer.write(final_image)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

            frame_count += 1

        writer.release()

        for listener in self.listeners:
            listener.finish(self.total_trackers_started, self.total_trackers_finished)
