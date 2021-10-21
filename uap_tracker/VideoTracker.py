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

    def create_trackers_from_keypoints(self, tracker_type, key_points, frame):
        for kp in key_points:
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

    def update_trackers(self, tracker_type, key_points, frame):

        # cache kp -> bbox mapping for removing failed trackers
        kp_bbox_map = {}
        for kp in key_points:
            bbox = u.kp_to_bbox(kp)
            kp_bbox_map[kp] = bbox

        failed_trackers = []
        for idx, tracker in enumerate(self.live_trackers):
            # Update tracker
            ok, bbox = tracker.cv2_tracker.update(frame)

            if ok:
                # Tracking success
                # tracker_success(tracker, bbox, masked)
                tracker.update_bbox(bbox)
            else:
                # Tracking failure
                failed_trackers.append(tracker)

            # Try to match the new detections with this tracker
            for idx, kp in enumerate(key_points):
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
                    self.create_and_add_tracker(tracker_type, frame, new_bbox)

    def detect_and_track(self, record=False, two_by_two=True):

        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'DASIAMRPN']
        background_subtractor_types = ['KNN']

        tracker_type = tracker_types[7]
        background_subtractor_type = background_subtractor_types[0]

        source_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Open output video
        writer = None
        if record:
            writer = u.get_writer("outputvideo.mp4", source_width, source_height) #  MWG: I don't like the idea of this being here, TODO Move this into a listener

        font_size = int(source_height / 1000.0)
        font_colour = (50, 170, 50)
        max_display_dim = 1080

        # Read first frame.
        ok, frame = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        background_subtractor = u.create_background_subtractor(background_subtractor_type)

        output_image, masked_background = u.apply_background_subtraction(image_gray, background_subtractor)

        for i in range(5):
            ok, frame = self.video.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output_image, masked_background = u.apply_background_subtraction(frame_gray, background_subtractor)

        key_points = u.perform_blob_detection(masked_background)

        # Create Trackers
        self.create_trackers_from_keypoints(tracker_type, key_points, output_image)

        frame_count = 0
        while True:
            # Read a new frame
            ok, frame = self.video.read()
            if not ok:
                break

            # Start timer
            timer = cv2.getTickCount()

            # Copy the frame as we want to mark the original and use the copy for displaying tracking artifacts
            image_frame = frame.copy()
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.putText(frame, 'Original Frame (Sky360)', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_colour, 2)

            # MG: This needs to be done on an 8 bit gray scale image, the colour image is causing a detection cluster
            output_image, masked_background = u.apply_background_subtraction(image_gray, background_subtractor)
            output_image = image_frame

            # Detect new objects of interest to pass to tracker
            key_points = u.perform_blob_detection(masked_background)

            self.update_trackers(tracker_type, key_points, output_image)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            for listener in self.listeners:
                listener.trackers_updated_callback(output_image, self.live_trackers, fps)

            for tracker in self.live_trackers:
                tracker.add_bbox_to_image(output_image, int(font_size/2), font_colour)

            msg = f"Trackers: started:{self.total_trackers_started}, ended:{self.total_trackers_finished}, alive:{len(self.live_trackers)}  (Sky360)"
            print(msg)
            cv2.putText(output_image, msg, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_colour, 2)
            cv2.putText(output_image, f"FPS: {str(int(fps))} (Sky360)", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_colour, 2)

            if two_by_two:
                # Create a copy as we need to put text on it and also turn it into a 24 bit image
                masked_background_copy = cv2.cvtColor(masked_background.copy(), cv2.COLOR_GRAY2BGR)

                masked_background_with_key_points = cv2.drawKeypoints(masked_background_copy, key_points, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                msg = f"Detected {len(key_points)} Key Points (Sky360)"
                cv2.putText(masked_background_with_key_points, msg, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_colour, 2)

                cv2.putText(masked_background_copy, "Masked Background (Sky360)", (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, font_colour, 2)

                im_h1 = cv2.hconcat([frame, output_image])
                im_h2 = cv2.hconcat([masked_background_copy, masked_background_with_key_points])

                final_image = cv2.vconcat([im_h1, im_h2])
            else:
                final_image = output_image

            # Display result, resize it to a standard size
            if final_image.shape[0] > max_display_dim or final_image.shape[1] > max_display_dim:
                #  MG: scale the image to something that is of a reasonable viewing size but write the original to file
                scaled_image = u.scale_image(final_image, max_display_dim)
                cv2.imshow("Tracking", scaled_image)
            else:
                cv2.imshow("Tracking", final_image)

            if writer is not None:
                writer.write(final_image)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

            frame_count += 1

        if writer is not None:
            writer.release()

        for listener in self.listeners:
            listener.finish(self.total_trackers_started, self.total_trackers_finished)
