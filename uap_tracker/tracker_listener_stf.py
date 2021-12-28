import os
import cv2
import uap_tracker.utils as utils
from uap_tracker.stf_writer import STFWriter
import json
import numpy as np
import uuid

#
# Listener to create output suitable for input to stage2
# in SimpleTrackerFormat (stf) format:
#
# ./<video_name>_<section_id>/
#   annotations.json
#   video.mp4
#   images/
#     <frame_id:06>.<image_contents>.jpg
#


class TrackerListenerStf():

    def __init__(self, video, file_name, output_dir,
                 zoom_level=10):
        self.video = video
        self.file_name = file_name
        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.zoom_level = zoom_level

    def _source_video_width_height(self):
        source_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return source_width, source_height

    def _target_video_width_height(self):
        pass

    def _create_stf_writer(self):
        width, height = self._target_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height)

    def finish(self):
        print(f"Finished processing {self.file_name}")


class TrackerListenerMOTStf(TrackerListenerStf):
    def __init__(self, video, file_name, output_dir):
        super().__init__(video, file_name, output_dir)

        self.stf_writer = None

    def _target_video_width_height(self):
        return self._source_video_width_height()

    def _mot(self, video_tracker):
        frame_id = video_tracker.get_frame_count()
        alive_trackers = video_tracker.get_live_trackers()
        frame = video_tracker.get_image(video_tracker.FRAME_TYPE_ORIGINAL)
        high_quality_trackers = map(lambda x: x.is_tracking(), alive_trackers)
        if sum(high_quality_trackers) > 0:

            if self.stf_writer is None:
                self.stf_writer = self._create_stf_writer()

            annotated_frame = video_tracker.get_annotated_image()
            for tracker in filter(lambda x: x.is_tracking(), alive_trackers):
                self.stf_writer.add_bbox(frame_id, tracker)

            self.stf_writer.write_original_frame(frame)
            self.stf_writer.write_annotated_frame(annotated_frame)
            self.stf_writer.write_images(
                video_tracker.get_images(), frame_id)

        else:
            self._close_segment()

    def _close_segment(self):
        if self.stf_writer:
            self.stf_writer.close()
            self.stf_writer = None

    def trackers_updated_callback(self, video_tracker):
        self._mot(video_tracker)

    def finish(self, total_trackers_started, total_trackers_finished):
        self._close_segment()
        super().finish()


class TrackerListenerSOTStf(TrackerListenerStf):
    def __init__(self, video, file_name, output_dir):
        super().__init__(video, file_name, output_dir)

        self.open_writers = {}

    def _target_video_width_height(self):
        source_width, source_height = self._source_video_width_height()
        return (
            int(source_width/self.zoom_level),
            int(source_height/self.zoom_level),
        )

    def _sot(self, video_tracker):
        frame_id = video_tracker.get_frame_count()
        alive_trackers = video_tracker.get_live_trackers()
        alive_trackers_to_process = set(
            filter(lambda x: x.is_tracking(), alive_trackers))
        processed = set()
        newly_closed = []
        for tracker, writer in self.open_writers.items():

            if tracker in alive_trackers_to_process:
                self.process_tracker(video_tracker, frame_id, tracker, writer)
            else:
                writer.close()
                newly_closed.append(tracker)
            processed.add(tracker)
        for tracker in newly_closed:
            self.open_writers.pop(tracker, None)
        for tracker in alive_trackers_to_process-processed:
            writer = self._create_stf_writer()
            self.open_writers[tracker] = writer
            self.process_tracker(video_tracker, frame_id, tracker, writer)

    def process_tracker(self, video_tracker, frame_id, tracker, writer):
        frame = video_tracker.get_image(video_tracker.FRAME_TYPE_ORIGINAL)
        writer.add_bbox(frame_id, tracker)
        zoom_frame = utils.zoom_and_clip(
            frame, tracker.get_center(), self.zoom_level)
        print(f"Zoom shape: {zoom_frame.shape}")
        writer.write_original_frame(zoom_frame)
        writer.write_images(video_tracker.get_images(), frame_id)

    def trackers_updated_callback(self, video_tracker):
        self._sot(video_tracker)

    def finish(self, total_trackers_started, total_trackers_finished):
        for _tracker, writer in self.open_writers.items():
            writer.close()
        super().finish()
