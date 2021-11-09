import cv2
import numpy as np

class TwoByTwoVisualiser():

    def __init__(self):
        self.font_size = None
        self.font_colour = None

    def initialise(self, font_size, font_colour):
        self.font_size = font_size
        self.font_colour = font_colour

    def visualise_frame(self, video_tracker, frame_input, frame_masked_background, frame_output, key_points, fps):

        cv2.putText(frame_input, 'Original Frame (Sky360)', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)

        msg = f"Trackers: trackable:{sum(map(lambda x: x.is_trackable(), video_tracker.live_trackers))}, alive:{len(video_tracker.live_trackers)}, started:{video_tracker.total_trackers_started}, ended:{video_tracker.total_trackers_finished} (Sky360)"
        print(msg)
        cv2.putText(frame_output, msg, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)
        cv2.putText(frame_output, f"FPS: {str(int(fps))} (Sky360)", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)

        # Create a copy as we need to put text on it and also turn it into a 24 bit image
        frame_masked_background_copy = cv2.cvtColor(frame_masked_background.copy(), cv2.COLOR_GRAY2BGR)

        frame_masked_background_with_key_points = cv2.drawKeypoints(frame_masked_background_copy, key_points, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        msg = f"Detected {len(key_points)} Key Points (Sky360)"
        cv2.putText(frame_masked_background_with_key_points, msg, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)

        cv2.putText(frame_masked_background_copy, "Masked Background (Sky360)", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)

        im_h1 = cv2.hconcat([frame_input, frame_output])
        im_h2 = cv2.hconcat([frame_masked_background_copy, frame_masked_background_with_key_points])

        return cv2.vconcat([im_h1, im_h2])
