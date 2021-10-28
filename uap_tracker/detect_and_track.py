#
# python uap_tracker/detect_and_track.py [filename]
#

#from typing_extensions import ParamSpecArgs
import cv2
import sys
from uap_tracker.video_tracker import VideoTracker

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
        input_file = "vids/uap_texas_skyhub.mp4"
        #input_file = "vids/Test_Trimmed.mp4"

    video = cv2.VideoCapture(input_file)
        
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    video_tracker = VideoTracker(video, VideoTracker.DETECTION_SENSITIVITY_HIGH)
    video_tracker.detect_and_track()
