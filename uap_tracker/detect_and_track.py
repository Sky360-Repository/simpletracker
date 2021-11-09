#
# python uap_tracker/detect_and_track.py [filename]
#

#from typing_extensions import ParamSpecArgs
import cv2
import sys
from uap_tracker.default_visualiser import DefaultVisualiser
from uap_tracker.two_by_two_visualiser import TwoByTwoVisualiser
from uap_tracker.video_tracker import VideoTracker
from uap_tracker.video_tracker_new import VideoTrackerNew
from video_playback_controller import VideoPlaybackController
from camera_stream_controller import CameraStreamController

#from object_detection.utils import config_util

if __name__ == '__main__':

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        #input_file = "vids/clouds_d6c9697b-d41e-4d72-9ed2-fc84fcec4ba5.mkv"
        #input_file = "vids/fast_bird_f7a83315-875f-4c04-af9e-2646a475f4da.mkv"
        #input_file = "vids/moving_clouds.mkv"
        #input_file = "vids/multiple_birds_f6aed5f0-c446-42d7-96e8-c5a37ef0e936.mkv"
        #input_file = "vids/rain_b29448a1-d491-4ba1-8e06-31a0699d9417.mkv"
        #input_file = "vids/uap_b253fd01-f670-4467-b200-6bbeac6649f0.mkv4"
        #input_file = "vids/uap_texas_skyhub.mp4"
        #input_file = "vids/Test_Trimmed.mp4"
        input_file = "vids/birds_and_plane.mp4"

    ##playback = VideoPlaybackController(input_file, visualiser=DefaultVisualiser())
    #playback = VideoPlaybackController(input_file, visualiser=TwoByTwoVisualiser())
    #playback.run(VideoTrackerNew.DETECTION_SENSITIVITY_HIGH, blur=True, normalise_video=True, mask_pct=92)

    playback = CameraStreamController()
    playback.run(VideoTrackerNew.DETECTION_SENSITIVITY_NORMAL, blur=True, normalise_video=True, mask_pct=92)