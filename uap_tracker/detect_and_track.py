#from typing_extensions import ParamSpecArgs
import cv2
import sys
import VideoTracker as vt

#from object_detection.utils import config_util

if __name__ == '__main__':

    # Read video
    #video = cv2.VideoCapture("videos/unknownclip.mp4") # the angles movement of uap
#    video = cv2.VideoCapture("videos/plane.mp4")
#    video = cv2.VideoCapture("videos/unknown_long.mp4")
#    video = cv2.VideoCapture("videos/unknown_direction_change.mp4")
#    video = cv2.VideoCapture("videos/samples/48301cdd-3947-43be-b0c1-01164f112625.mkv") # detecting alot of fast moving cloud (error)
#    video = cv2.VideoCapture("videos/samples/a994adff-8867-41e8-8241-6814257f202d.mkv") # detecting rain error
#    video = cv2.VideoCapture("videos/samples/577f38f5-de96-4928-af7c-121c3a78f1b1.mkv")# ok empty
#    video = cv2.VideoCapture("videos/samples/edb5ef90-62cf-4f6e-9e6c-e5409effddb3.mkv") #ok empty sky
#    video = cv2.VideoCapture("videos/samples/51201f30-75da-4c3f-bb1d-893f9a28ff68.mkv") #black bird, misses bird black??
#    video = cv2.VideoCapture("videos/samples/9e9bb2e5-4fd7-46ab-be98-0cd67f455ece.mkv") #rain, ok empty
#    video = cv2.VideoCapture("videos/samples/edd6edf4-96d0-45c8-a9cc-bf23a295e55a.mkv") #good, nothing no tracks
#    video = cv2.VideoCapture("videos/samples/a618e162-fc97-46fb-9eb5-f91e5cb6cc47.mkv") #nothing
#    video = cv2.VideoCapture("videos/samples/a683e74a-489e-4eb7-b443-fa9cc97b94d8.mkv") #nothing
#    video = cv2.VideoCapture("videos/samples/a994adff-8867-41e8-8241-6814257f202d.mkv")
#    video = cv2.VideoCapture("videos/samples/a150066f-f2ee-477e-b0ba-e8b49fb70127.mkv")
#    video = cv2.VideoCapture("videos/samples/aaaddcc6-687d-4483-9ce0-ac63211db7ef.mkv")
#    video = cv2.VideoCapture("videos/samples/aca44dd0-fdb3-4bc1-a76f-c3d85831ca88.mkv") # crane moving
#    video = cv2.VideoCapture("videos/samples/ae0e4eeb-c5c5-47bf-bd36-84a263a78e8c.mkv")
#    video = cv2.VideoCapture("videos/pelican.mp4")
#    video = cv2.VideoCapture("videos/cloud_plane.mp4")
# '03b53a8a-b5a0-4192-834e-48f2f56c007a.mkv' #insect
    input_file = "../videos/unknown_long.mp4"

    video = cv2.VideoCapture(input_file)
        
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    video_tracker = vt.VideoTracker(video)
    video_tracker.detect_and_track()


