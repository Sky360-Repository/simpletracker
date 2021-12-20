import cv2
import sys
from uap_tracker.video_playback_controller import VideoPlaybackController
from uap_tracker.frame_processor import FrameProcessor

class VideoPlaybackControllerCuda(VideoPlaybackController):

    def run(self):

        if not self.capture.isOpened():
            print(f"Could not open video stream")
            sys.exit()

        frame_count = 0
        fps = 0
        while cv2.waitKey(1) != 27:  # Escape
            success, frame = self.capture.read()
            if success:

                timer = cv2.getTickCount()

                with FrameProcessor.GPU(frame, self.video_tracker.dof_cuda) as processor:
                    self.video_tracker.process_frame_cuda(processor, frame, frame_count, fps)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                frame_count += 1
            else:
                break

        self.video_tracker.finalise()
