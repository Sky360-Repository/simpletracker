# Original work Copyright (c) 2022 Sky360
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

import os
import cv2
import json
import shutil
import uap_tracker.utils as utils
import time

class STFWriter():

    video_count = 0
    training_count = 0

    @classmethod
    def _get_and_increment_video_count(cls):
        ret = cls.video_count
        cls.video_count += 1
        return ret

    @classmethod
    def _get_and_increment_training_count(cls):
        ret = cls.training_count
        cls.training_count += 1
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
        self.training_dir = os.path.join(self.final_video_dir, 'training')
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
        x1, y1, w, h = utils.get_sized_bbox_from_tracker(tracker)
        print(f"{tracker.id}, {(x1, y1, w, h)}")
        self._add_trackid_label(tracker.id, 'unknown')
        return {
            'bbox': (x1, y1, w, h),
            'track_id': tracker.id,
            'timestamp' : time.time()
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

    def write_training(self, frame, frame_id, tracker):
        size_increment = 32

        if not os.path.isdir(self.training_dir):
            os.mkdir(self.training_dir)

        if tracker.is_tracking():
            x, y, w, h = tracker.get_bbox()

            # ensure we are dealing with even numbers
            if x % 2 != 0: x = x+1
            if y % 2 != 0: y = y+1

            # most training images will be 32 x 32 but just in case tracked objects are larger, account for that
            for i in range(1, 10):

                loop_increment = size_increment * i

                margin_w_tot = loop_increment - w
                margin_h_tot = loop_increment - h                

                if margin_w_tot > 0 and margin_h_tot > 0:
                    margin_w = int(margin_w_tot/2)
                    margin_h = int(margin_h_tot/2)
                    filename = os.path.join(self.training_dir, f"{frame_id:06}.{tracker.id}.{self._get_and_increment_training_count():}.{loop_increment}x{loop_increment}.jpg")
                    #print(f"write training image: {filename} - ({x},{y},{w},{h}) -> ({x},{y},{loop_increment},{loop_increment})")
                    training_img = frame[y-margin_h:y+h+margin_h, x-margin_w:x+w+margin_w]
                    cv2.imwrite(filename, training_img)
                    break

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
