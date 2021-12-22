import os
import cv2
import json
import shutil
import uap_tracker.utils as utils

class STFWriter():

    video_count = 0

    @classmethod
    def _get_and_increment_video_count(cls):
        ret = cls.video_count
        cls.video_count += 1
        return ret

    def __init__(self,
                 stf_output_dir,
                 video_file_root_name,
                 source_width,
                 source_height,
                 video_name='video.mp4',
                 movement_alpha=True,
                 annotate=True):

        self.annotate = annotate
        self.video_id = -1
        if annotate:
            self.video_id = self._get_and_increment_video_count()

        self.writer = None
        self.annotated_writer = None

        self.video_dir = None
        self.video_filename = None
        self.annotated_video_filename = None

        self.source_width = source_width
        self.source_height = source_height

        self.final_video_dir = stf_output_dir
        if annotate:
            self.final_video_dir = os.path.join(
                stf_output_dir, f"{video_file_root_name}_{self.video_id:06}")

        if not os.path.isdir(self.final_video_dir):
            os.mkdir(self.final_video_dir)

        self.annotations = {
            'track_labels': {},
            'frames': []
        }

        self.movement_alpha = movement_alpha

        ###

        self.images_dir = os.path.join(self.final_video_dir, 'images')
        self.video_filename = os.path.join(self.final_video_dir, video_name)
        self.annotated_video_filename = os.path.join(self.final_video_dir, "annotated_{0}".format(video_name))

    def _close_video_writers(self):
        self.writer.release()
        self.writer = None
        if self.annotated_writer:
            self.annotated_writer.release()
            self.annotated_writer = None

    def _add_trackid_label(self, track_id, label):
        if track_id not in self.annotations['track_labels']:
            self.annotations['track_labels'][track_id] = label

    def _create_stf_annotation(self, tracker):
        print(f"{tracker.id}, {tracker.get_bbox()}")
        self._add_trackid_label(tracker.id, 'unknown')
        return {
            'bbox': tracker.get_bbox(),
            'track_id': tracker.id
        }

    def add_bbox(self, frame_id, tracker):
        last_frame = None
        if len(self.annotations['frames']) > 0:
            last_frame = self.annotations['frames'][-1]
            if last_frame['frame'] != frame_id:
                last_frame = None

        if not last_frame:
            last_frame = {
                'frame': frame_id,
                'annotations': []
            }
            self.annotations['frames'].append(last_frame)

        last_frame['annotations'].append(self._create_stf_annotation(tracker))

    def write_original_frame(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]
        if not self.writer:
            self.writer = utils.get_writer(
                self.video_filename, width, height)
        self.writer.write(frame)

    def write_annotated_frame(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]
        if not self.annotated_writer:
            self.annotated_writer = utils.get_writer(
                self.annotated_video_filename, width, height)
        self.annotated_writer.write(frame)

    def write_images(self, images, frame_id):
        if not os.path.isdir(self.images_dir):
            os.mkdir(self.images_dir)
        for name, image in images.items():
            filename = os.path.join(
                self.images_dir, f"{frame_id:06}.{name}.jpg")
            cv2.imwrite(filename, image)

    def _close_annotations(self):
        filename = os.path.join(self.final_video_dir, 'annotations.json')
        with open(filename, 'w') as outfile:
            json.dump(self.annotations, outfile, indent=2)

    def close(self, min_annotations=25):
        print(
            f"close segment called on {self.final_video_dir} {self.writer}, min_annotations:{min_annotations}")
        if self.writer:
            self._close_video_writers()

            if self.annotate:
                # only save if >= 5 frames
                print(self.annotations)
                if len(self.annotations['frames']) >= min_annotations:
                    self._close_annotations()
                else:
                    shutil.rmtree(self.final_video_dir)
