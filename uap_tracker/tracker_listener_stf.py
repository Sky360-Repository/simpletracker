import os
import cv2
import uap_tracker.utils as utils
import json
import numpy as np
import shutil
import uuid

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

class STFWriter():
    
    video_count=0
    @classmethod
    def _get_and_increment_video_count(cls):
        ret=cls.video_count
        cls.video_count+=1
        return ret
   
    def __init__(self, 
                 stf_output_dir, 
                 video_file_root_name, 
                 source_width, 
                 source_height, 
                 movement_alpha=True):

        self.video_id=self._get_and_increment_video_count()
                
        self.writer = None
        self.annotated_writer = None

        self.video_dir=None
        self.video_filename=None
        self.annotated_video_filename = None

        self.final_video_dir=f"{stf_output_dir}/{video_file_root_name}_{self.video_id:06}"

        self.annotations={
            'track_labels':{},
            'frames':[]
        }

        self.movement_alpha=movement_alpha

        ###

        tmp_filename = str(uuid.uuid4())

        self.tmp_video_dir=os.path.join(stf_output_dir,tmp_filename)
        os.mkdir(self.tmp_video_dir)
        
        self.images_dir = self.tmp_video_dir + '/images/'
        os.mkdir(self.images_dir)

        self.video_filename = self.tmp_video_dir + '/' + 'video.mp4'
        self.annotated_video_filename = self.tmp_video_dir + '/' + 'annotated_video.mp4'
        
        self.writer = utils.get_writer(self.video_filename, source_width, source_height)
        self.annotated_writer = utils.get_writer(self.annotated_video_filename, source_width, source_height)

    def _close_video_writers(self):
        self.writer.release()
        self.writer=None
        self.annotated_writer.release()
        self.annotated_writer = None
    
    def _add_trackid_label(self, track_id, label):
        if track_id not in self.annotations['track_labels']:
            self.annotations['track_labels'][track_id]=label
            
    def _create_stf_annotation(self, tracker):
        print(f"{tracker.id}, {tracker.get_bbox()}")

        self._add_trackid_label(tracker.id, 'unknown')

        return {
            'bbox':tracker.get_bbox(),
            'track_id':tracker.id
        }

    def add_bbox(self, frame_id ,tracker):
        last_frame=None
        if len(self.annotations['frames']) > 0:
            last_frame=self.annotations['frames'][-1]
            if last_frame['frame']!=frame_id:
                last_frame=None

        if not last_frame:
            last_frame={
                'frame':frame_id,
                'annotations': []
            }
            self.annotations['frames'].append(last_frame)

        last_frame['annotations'].append(self._create_stf_annotation(tracker))
    

    def write_original_frame(self, frame):
        self.writer.write(frame)

    def write_annotated_frame(self, frame):
        self.annotated_writer.write(frame)

    def write_image(self, frame, frame_gray, frame_masked_background,frame_id):
        filename = self.images_dir + f"{frame_id:06}.png"
        if self.movement_alpha:
            # First create the image with alpha channel
            c_red, c_green, c_blue = cv2.split(frame)
            image = cv2.merge((c_red, c_green, c_blue,frame_masked_background))
        else:
            zero_channel = np.zeros(frame_gray.shape, dtype="uint8")
            image=cv2.merge([frame_gray, zero_channel, frame_masked_background])
        #print(f"writing {image.shape} to {filename}")
        cv2.imwrite(filename,image)

    def _close_annotations(self):
        
        filename=os.path.join(self.tmp_video_dir,'annotations.json')

        with open(filename, 'w') as outfile:
            json.dump(self.annotations, outfile, indent=2)

    def close(self):
        print(f"close segment called on {self.final_video_dir} {self.writer}")
        if self.writer:
            self._close_video_writers()
            #only save if >= 5 frames
            print(self.annotations)
            if len(self.annotations['frames']) >= 25:
                self._close_annotations()
                print(f"Renaming {self.tmp_video_dir} to {self.final_video_dir}")
                os.rename(self.tmp_video_dir,self.final_video_dir)
            else:
                shutil.rmtree(self.tmp_video_dir)



class TrackerListenerStf():

    def __init__(self, video, full_path, file_name, output_dir, zoom_level=10):
        self.video = video
        self.full_path = full_path
        self.file_name = file_name
        self.output_dir = output_dir
        self.recording = False
        self._create_output_dir('/stf')
        self.stf_dir = self._create_output_dir('/stf/')
        self.processed_dir = self._create_output_dir('/processed/')
        self.zoom_level=zoom_level

        
    def _create_output_dir(self, dir_ext):
        dir_to_create = self.output_dir + dir_ext
        if not os.path.isdir(dir_to_create):
            os.mkdir(dir_to_create)
        return dir_to_create

    def _source_video_width_height(self):
        source_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return source_width, source_height

    def _target_video_width_height(self):
        pass

    def _create_stf_writer(self):
        width, height = self._target_video_width_height()
        return STFWriter(self.stf_dir, self.file_name, width, height)    

    def initialise(self, sensitivity, blur, normalise_video, tracker_type, background_subtractor_type, source_width, source_height):
        pass

    def finish(self):
        os.rename(self.full_path, self.processed_dir + os.path.basename(self.full_path))


class TrackerListenerMOTStf(TrackerListenerStf):
    def __init__(self, video, full_path, file_name, output_dir):
        super().__init__(video, full_path, file_name, output_dir)

        self.stf_writer=None

    def _target_video_width_height(self):
        return self._source_video_width_height()

    def _mot(self, frame, frame_gray, frame_masked_background, frame_id, alive_trackers):
        high_quality_trackers=map(lambda x : x.is_trackable(), alive_trackers)
        if sum(high_quality_trackers) > 0:

            if self.stf_writer is None:
                self.stf_writer = self._create_stf_writer()

            annotated_frame = frame.copy()
            for tracker in filter(lambda x : x.is_trackable(),alive_trackers):
                self.stf_writer.add_bbox(frame_id,tracker)
                utils.add_bbox_to_image(tracker.get_bbox(), annotated_frame, tracker.id, 1, (0, 255, 0))

            self.stf_writer.write_original_frame(frame)
            self.stf_writer.write_annotated_frame(annotated_frame)
            self.stf_writer.write_image(frame, frame_gray, frame_masked_background, frame_id)

        else:            
            self._close_segment()

    def _close_segment(self):
        if self.stf_writer:
            self.stf_writer.close()
            self.stf_writer=None

    def trackers_updated_callback(self, frame, frame_gray, frame_masked_background, frame_id, alive_trackers, fps):
        self._mot(frame, frame_gray, frame_masked_background, frame_id, alive_trackers)

    def finish(self, total_trackers_started, total_trackers_finished):
        self._close_segment()
        super().finish()


class TrackerListenerSOTStf(TrackerListenerStf):
    def __init__(self, video, full_path, file_name, output_dir):
        super().__init__(video, full_path, file_name, output_dir)

        self.open_writers={}

    def _target_video_width_height(self):
        source_width, source_height = self._source_video_width_height()
        return (
            int(source_width/self.zoom_level),
            int(source_height/self.zoom_level),
        )

    def _sot(self, frame, frame_gray, frame_masked_background, frame_id, alive_trackers):
        high_quality_trackers=map(lambda x : x.is_trackable(), alive_trackers)
                
        alive_trackers_to_process=set(filter(lambda x : x.is_trackable(),alive_trackers))
        processed=set()
        newly_closed=[]
        for tracker,writer in self.open_writers.items():
            if tracker in alive_trackers_to_process:
                self.process_tracker(frame, frame_gray, frame_masked_background, frame_id, tracker, writer)
            else:
                writer.close()
                newly_closed.append(tracker)
            processed.add(tracker)
        for tracker in newly_closed:
            self.open_writers.pop(tracker, None)
        for tracker in alive_trackers_to_process-processed:
            writer=self._create_stf_writer()
            self.open_writers[tracker] = writer
            self.process_tracker(frame, frame_gray, frame_masked_background, frame_id, tracker, writer)


    def process_tracker(self, frame, frame_gray, frame_masked_background, frame_id, tracker, writer):
        writer.add_bbox(frame_id,tracker)
        zoom_frame=utils.zoom_and_clip(frame, tracker.get_center(), self.zoom_level)
        print(f"Zoom shape: {zoom_frame.shape}")
        writer.write_original_frame(zoom_frame)
        
        annotated_frame = zoom_frame.copy()
        utils.add_bbox_to_image(tracker.get_bbox(), annotated_frame, tracker.id, 1, (0, 255, 0))                
        writer.write_annotated_frame(annotated_frame)
                
        writer.write_image(frame, frame_gray, frame_masked_background, frame_id)

    def trackers_updated_callback(self, frame, frame_gray, frame_masked_background, frame_id, alive_trackers, fps):
        self._sot(frame, frame_gray, frame_masked_background, frame_id, alive_trackers)

    def finish(self, total_trackers_started, total_trackers_finished):
        for _tracker,writer in self.open_writers.items():
            writer.close()
        super().finish()


