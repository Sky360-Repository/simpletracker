import os
import cv2
import uap_tracker.utils as u
import json

#
# Listener to create supervise.ly video format output
#
class TrackerListenerStf():

    def __init__(self, video, full_path, file_name, output_dir):
        self.video = video
        self.full_path = full_path
        self.file_name = file_name
        self.output_dir = output_dir
        self.recording = False
        self._create_output_dir('/stf')
        self.stf_dir = self._create_output_dir('/stf/')
        self.processed_dir = self._create_output_dir('/processed/')

        self.video_id=0
        self.writer = None
        self.video_dir=None
        self.video_filename=None

        self.frame_annotations=[]

        print(f"TrackerListenerSly processing {full_path}")

    def _create_output_dir(self, dir_ext):
        dir_to_create = self.output_dir + dir_ext
        if not os.path.isdir(dir_to_create):
            os.mkdir(dir_to_create)
        return dir_to_create

    def trackers_updated_callback(self, frame, frame_id, alive_trackers, fps):
        if len(alive_trackers) > 0:
            if self.writer is None:
                self._init_writer()
            
            for tracker in alive_trackers:
                self.frame_annotations.append({
                    'frame':frame_id,
                    'annotations': self._create_stf_annotation(frame_id, tracker)
                })
                self._write_image(frame,frame_id)

            
            self.writer.write(frame)
        else:
            if self.writer:
                # No more live trackers, so close out this video
                self._close_writer()
                self._close_annotations()

    def _write_image(self,frame,frame_id):
        filename = self.images_dir + f"{frame_id:06}.jpg"
        cv2.imwrite(filename,frame)


    def _create_stf_annotation(self,frame_id, tracker):
        print(f"{tracker.id}, {tracker.get_bbox()}")

        return {
            'bbox':tracker.get_bbox(),
            'track_id':tracker.id,
            'class':'unknown'
        }

    def finish(self, total_trackers_started, total_trackers_finished):
        if self.writer:
            self._close_writer()
            self._close_annotations()
        os.rename(self.full_path, self.processed_dir + os.path.basename(self.full_path))


    def _init_writer(self):
        self.video_dir=f"{self.stf_dir}/{self.file_name}_{self.video_id:06}"
        os.mkdir(self.video_dir)
        self.video_id += 1

        self.images_dir = self.video_dir + '/images/'
        os.mkdir(self.images_dir)     

        self.video_filename = self.video_dir + '/' + 'video.mp4'

        source_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.writer = u.get_writer(self.video_filename, source_width, source_height)


    def _close_writer(self):
        self.writer.release()
        self.writer=None

    def _close_annotations(self):
        filename=self.video_dir + '/annotations.json'
        with open(filename, 'w') as outfile:
            json.dump(self.frame_annotations, outfile, indent=2)
        self.frame_annotations=[]