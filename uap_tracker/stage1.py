# usage: python uap_tracker/stage1.py -i <inputdir> -o <outputdirectory>
# e.g. python uap_tracker/stage1.py -i videos/samples/ -o videos/sp3

import os
import getopt
import sys

import cv2

import detect_and_track

USAGE='python uap_tracker/stage1.py -i <inputdir> -o <outputdirectory>'

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:o:p:",["ifile=","ofile=","pfile="])
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

        full_path=input_dir+filename

        video =  cv2.VideoCapture(full_path)
        
        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()
        
        video_tracker=detect_and_track.VideoTracker(video)

        listener=TrackerListener(video, full_path, root_name, output_dir)
        
        video_tracker.listen(listener)
        video_tracker.detect_and_track()




class TrackerListener():
    def __init__(self,video,full_path, file_name, output_dir):
        self.video=video
        self.full_path=full_path
        self.file_name=file_name
        self.output_dir=output_dir
        self.recording=False
        
        self.movement_dir=self._create_output_dir('movement/')
        self.no_movement_dir=self._create_output_dir('no_movement/')
        self.transcoded_dir=self._create_output_dir('transcoded/')

        self.writer_filename=f"{self.transcoded_dir}{self.file_name}.mp4"
        self.writer=None
        
        print(f"TrackerListener processing {full_path}")

    def _create_output_dir(self,dir_ext):
        dir_to_create=self.output_dir+dir_ext
        if not os.path.isdir(dir_to_create):
            os.mkdir(dir_to_create)
        return dir_to_create
        


    def trackers_updated_callback(self,frame, alive_trackers, fps):

        if self.recording or len(alive_trackers)>0:
            self.recording=True
            if self.writer is None:
                self._init_writer()
            for tracker in alive_trackers :
                tracker.add_bbox_to_image(frame,(0,255,0))
            self.writer.write(frame)
          
    def finish(self,total_trackers_started,total_trackers_finished):
        if self.writer:
            self.writer.release()
            print(f"TrackerListener detected movement in {self.full_path} transcoded to {self.writer_filename}")
            os.rename(self.full_path, self.movement_dir+os.path.basename(self.full_path))
        else:
            print(f"TrackerListener no movement detected in {self.full_path}")
            os.rename(self.full_path, self.no_movement_dir+os.path.basename(self.full_path))
            

    def _init_writer(self):
        source_width=int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height=int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer=detect_and_track.get_writer(self.writer_filename,source_width,source_height)


    
    


if __name__ == '__main__' :
    main(sys.argv[1:])
