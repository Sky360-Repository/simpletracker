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
        frame = np.empty((1024, 1024, 3),np.uint8)
        frame_grey = np.empty((1024, 1024, 3),np.uint8)
        frame_masked_background = np.empty((1024, 1024, 3),np.uint8)
        keypoints = []

        with FrameProcessor.GPU(
                resize_frame=self.video_tracker.resize_frame,
                noise_reduction=self.video_tracker.noise_reduction,
                mask_pct=self.video_tracker.mask_pct,
                detection_mode=self.video_tracker.detection_mode,
                detection_sensitivity=self.video_tracker.detection_sensitivity,
                blur_radius=self.video_tracker.blur_radius) as processor:
            while cv2.waitKey(1) != 27:  # Escape
                success, frame = self.capture.read()
                if success:
                    timer = cv2.getTickCount()
                    self.video_tracker.process_frame(processor, frame, frame_grey, frame_masked_background, keypoints, frame_count, fps, stream=cv2.cuda.Stream_Null())
                    # Calculate Frames per second (FPS)
                    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                    frame_count += 1
                else:
                    break
            self.video_tracker.finalise()
