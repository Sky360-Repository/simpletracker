from uap_tracker import utils
from uap_tracker.tracker import Tracker


class Visualizer():
    def __init__(self, max_display_dim):
        self.max_display_dim = max_display_dim
        self.font_size = max(1, self.max_display_dim/1000)
        self.font_colour = (50, 170, 50)

    def display(self, frame):
        # Display result, resize it to a standard size
        utils.display_frame(frame, self.max_display_dim)

    def Visualize(self, video_tracker):
        msg = f"Trackers: trackable:{sum(map(lambda x: x.is_tracking(), video_tracker.live_trackers))}, alive:{len(video_tracker.live_trackers)}, started:{video_tracker.total_trackers_started}, ended:{video_tracker.total_trackers_finished}, {video_tracker.get_fps()}fps (Sky360)"
        print(msg)
        frame_output = self.visualise_frame(video_tracker)
        if frame_output is not None:
            self.display(frame_output)
