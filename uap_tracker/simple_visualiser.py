import cv2
from visualizer import Visualizer


class SimpleVisualiser(Visualizer):
        
    def visualise_frame(self, video_tracker):

        frame_output = video_tracker.get_image('original').copy()
        fps = video_tracker.get_fps()
        msg = f"Trackers: trackable:{sum(map(lambda x: x.is_trackable(), video_tracker.live_trackers))}, alive:{len(video_tracker.live_trackers)}, started:{video_tracker.total_trackers_started}, ended:{video_tracker.total_trackers_finished} (Sky360)"
        print(msg)
        cv2.putText(frame_output, msg, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)
        cv2.putText(frame_output, f"FPS: {str(int(fps))} (Sky360)", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_colour, 2)

        return frame_output
