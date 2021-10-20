# usage: python uap_tracker/stage1.py -i <inputdir> -o <outputdirectory>
# e.g. python uap_tracker/stage1.py -i videos/samples/ -o videos/sp3

import os
import getopt
import sys
import cv2
import VideoTracker as vt
import TrackerListener as tl

USAGE = 'python uap_tracker/stage1.py -i <inputdir> -o <outputdirectory>'

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:o:p:", ["ifile=", "ofile=", "pfile="])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ("-i", "--idir"):
            input_dir = arg
        elif opt in ("-o", "--odir"):
            output_dir = arg

    print('Input dir is ', input_dir)
    print('Output dir is ', output_dir)

    for filename in os.listdir(input_dir):

        base = os.path.basename(filename)
        root_name = os.path.splitext(base)[0]

        full_path = input_dir + filename

        video = cv2.VideoCapture(full_path)

        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        video_tracker = vt.VideoTracker(video)

        listener = tl.TrackerListener(video, full_path, root_name, output_dir)

        video_tracker.listen(listener)
        video_tracker.detect_and_track()

if __name__ == '__main__':
    main(sys.argv[1:])
