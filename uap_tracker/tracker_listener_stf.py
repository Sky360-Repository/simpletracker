import os
import cv2
import uap_tracker.utils as utils
import json
import numpy as np
import shutil

#
# Listener to create output suitable for input to stage2
# in SimpleTrackerFormat (stf) format:
#
# ./processed/
# ./stf/<video_name>_<section_id>/
#   annotations.json
#   video.mp4
#   images/
#     <frame_id:06>.jpg
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
        self.annotated_writer = None

        self.video_dir=None
        self.video_filename=None
        self.annotated_video_filename = None

        self.frame_annotations=[]

        self.labels={}

        self.final_dir=None

        print(f"TrackerListenerSly processing {full_path}")

    def _create_output_dir(self, dir_ext):
        dir_to_create = self.output_dir + dir_ext
        if not os.path.isdir(dir_to_create):
            os.mkdir(dir_to_create)
        return dir_to_create

    def trackers_updated_callback(self, frame, frame_gray, frame_masked_background, frame_id, alive_trackers, fps):
        if len(alive_trackers) > 0:
            if self.writer is None:
                self._init_writer()
            
            frame_annotations={
                    'frame':frame_id,
                    'annotations': []
            }
            for tracker in alive_trackers:
                frame_annotations['annotations'].append(self._create_stf_annotation(frame_id, tracker))
            self.frame_annotations.append(frame_annotations)
            self._write_image(frame_gray, frame_masked_background, frame_id)

            self.writer.write(frame)

            annotated_frame = frame.copy()
            utils.add_bbox_to_image(tracker.get_bbox(), annotated_frame, tracker.id, 1, (0, 255, 0))
            self.annotated_writer.write(annotated_frame)
        else:
            if self.writer:
                self._close_segment()

    def _write_image(self,frame_gray, frame_masked_background,frame_id):
        filename = self.images_dir + f"{frame_id:06}.jpg"
        zero_channel = np.zeros(frame_gray.shape, dtype="uint8")
        image=cv2.merge([frame_gray, zero_channel, frame_masked_background])
        cv2.imwrite(filename,image)

    def _add_trackid_label(self, track_id, label):
        if track_id not in self.labels:
            self.labels[track_id]=label

    def _create_stf_annotation(self,frame_id, tracker):
        print(f"{tracker.id}, {tracker.get_bbox()}")

        self._add_trackid_label(tracker.id, 'unknown')

        return {
            'bbox':tracker.get_bbox(),
            'track_id':tracker.id
        }

    def _close_segment(self):
        if self.writer:
            self._close_writer()
            #only save if >= 5 frames
            if len(self.frame_annotations) >= 5:
                self._close_annotations()
                final_video_dir=f"{self.stf_dir}/{self.file_name}_{self.video_id:06}"
                self.video_id += 1
                print(f"Renaming {self.tmp_video_dir} to {final_video_dir}")
                os.rename(self.tmp_video_dir,final_video_dir)
            else:
                shutil.rmtree(self.tmp_video_dir)

    def finish(self, total_trackers_started, total_trackers_finished):
        self._close_segment()
        os.rename(self.full_path, self.processed_dir + os.path.basename(self.full_path))

    def _init_writer(self):
        self.tmp_video_dir=f"{self.stf_dir}/tmp"
        os.mkdir(self.tmp_video_dir)
        
        
        self.images_dir = self.tmp_video_dir + '/images/'
        os.mkdir(self.images_dir)     

        self.video_filename = self.tmp_video_dir + '/' + 'video.mp4'
        self.annotated_video_filename = self.tmp_video_dir + '/' + 'annotated_video.mp4'

        source_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.writer = utils.get_writer(self.video_filename, source_width, source_height)
        self.annotated_writer = utils.get_writer(self.annotated_video_filename, source_width, source_height)

    def _close_writer(self):
        self.writer.release()
        self.writer=None
        self.annotated_writer.release()
        self.annotated_writer = None

    def _close_annotations(self):
        
        filename=self.tmp_video_dir + '/annotations.json'

        annotations={
            'track_labels':self.labels,
            'frames':self.frame_annotations
        }

        with open(filename, 'w') as outfile:
            json.dump(annotations, outfile, indent=2)
        
        self.frame_annotations=[]
