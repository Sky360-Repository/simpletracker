import cv2
import os
import sys


def get_camera(config):
    camera_mode = config.get('mode', 'rtsp')
    camera_uri = config.get('camera_uri',
                            'rtsp://admin:sky360@192.168.1.108:554/live')
    print(f"Connecting to {camera_mode} camera at {camera_uri}")
    camera = None
    if camera_mode == 'ffmpeg':
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        camera = cv2.VideoCapture(
            camera_uri,
            cv2.CAP_FFMPEG
        )
    elif camera_mode == 'rtsp':
        camera = cv2.VideoCapture(
            camera_uri
        )
    elif camera_mode == 'local':
        for i in range(-1, 100):
            try:
                camera = cv2.VideoCapture(i)
                if camera:
                    break
            except cv2.error as e:
                print(e)
            except Exception as e:
                print(e)
    ##
    # Apparently this works for firewire, but I couldn't get it to
    ## camera = cv2.VideoCapture(3, cv2.CAP_DC1394)

    if not camera:
        print(cv2.getBuildInformation())
        print(f"Unable to find camera using config: {config}")
        sys.exit(1)
    return camera
