from visualizer import Visualizer


class NoOpVisualiser(Visualizer):

    def visualise_frame(self, video_tracker):
        return None
