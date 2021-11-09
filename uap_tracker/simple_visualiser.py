import cv2

class SimpleVisualiser():

    def __init__(self):
        self.font_size = None
        self.font_colour = None

    def initialise(self, font_size, font_colour):
        self.font_size = font_size
        self.font_colour = font_colour

    def visualise_frame(self, video_tracker, frame_input, frame_masked_background, frame_output, key_points, fps):

        msg = f"Trackers: trackable:{sum(map(lambda x: x.is_trackable(), video_tracker.live_trackers))}, alive:{len(video_tracker.live_trackers)}, started:{video_tracker.total_trackers_started}, ended:{video_tracker.total_trackers_finished} (Sky360)"
        print(msg)
        cv2.putText(frame_output, msg, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)
        cv2.putText(frame_output, f"FPS: {str(int(fps))} (Sky360)", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)

        return frame_output