from uap_tracker import utils
from uap_tracker.tracker import Tracker


class Visualizer():
    def __init__(self):
        self.max_display_dim = 1080
        self.font_size = max(1, self.max_display_dim/250)
        self.font_colour = (50, 170, 50)

    def display(self, frame):
        # Display result, resize it to a standard size
        utils.display_frame(frame, self.max_display_dim)

    # Event Listener API

    def trackers_updated_callback(self, video_tracker):
        frame_output = self.visualise_frame(video_tracker)

        self.display(frame_output)

    def finish(self, total_trackers_started, total_trackers_finished):
        pass

    # End Event Listener API
