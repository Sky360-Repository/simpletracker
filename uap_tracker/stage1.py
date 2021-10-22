# usage: python uap_tracker/stage1.py -i <inputdir> -o <outputdirectory>
# e.g. python uap_tracker/stage1.py -i videos/samples/ -o videos/sp3

import os
import getopt
import sys
import cv2
import VideoTracker as vt
import TrackerListener as tl
import TrackerListenerStf

USAGE = 'python uap_tracker/stage1.py -i <inputdir> -o <outputdirectory> [-f [original|stf]]'

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:o:f:", ["ifile=", "ofile=", "format="])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)

    #default output format to simpletracker
    format = 'original'

    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ("-i", "--idir"):
            input_dir = arg
        elif opt in ("-o", "--odir"):
            output_dir = arg
        elif opt in ("-f", "--format"):
            format = arg

    print('Input dir is ', input_dir)
    print('Output dir is ', output_dir)
    print('Format is ', format)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for filename in os.listdir(input_dir):

        base = os.path.basename(filename)
        root_name = os.path.splitext(base)[0]

        full_path = input_dir + filename

        print(f"Opening {full_path}")
        video = cv2.VideoCapture(full_path)

        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        video_tracker = vt.VideoTracker(video)

        if format=='original':
            clz=tl.TrackerListener
        elif format=='stf':
            clz=TrackerListenerStf.TrackerListenerStf
        
        listener = clz(video, full_path, root_name, output_dir)

        video_tracker.listen(listener)
        video_tracker.detect_and_track()

if __name__ == '__main__':
    main(sys.argv[1:])
