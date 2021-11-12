import cv2

class DefaultVisualiser():

    def __init__(self):
        self.font_size = None
        self.font_colour = None

    def initialise(self, font_size, font_colour):
        self.font_size = font_size
        self.font_colour = font_colour

    def visualise_frame(self, video_tracker, frame_input, frame_masked_background, frame_output, key_points, fps):

        msg = f"Trackers: trackable:{sum(map(lambda x: x.is_trackable(), video_tracker.live_trackers))}, alive:{len(video_tracker.live_trackers)}, started:{video_tracker.total_trackers_started}, ended:{video_tracker.total_trackers_finished} (Sky360)"
        print(msg)

        return frame_output
