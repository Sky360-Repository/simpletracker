import os
import cv2
from datetime import datetime, timedelta

from tracker_listener_stf import STFWriter
from video_formatter import VideoFormatter

class OriginalFrameDumper(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='original_frames.mp4')

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()
        self.writer.write_original_frame(video_tracker.get_image('original'))

class GreyFrameDumper(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='grey_frames.mp4')

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()
        self.writer.write_original_frame(video_tracker.get_image('grey'))

class OpticalFlowFrameDumper(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='optical_flow_frames.mp4')

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()
        self.writer.write_original_frame(video_tracker.get_image('optical_flow'))

class AnnotatedFrameDumper(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='anotated_frames.mp4')

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()
        self.writer.write_original_frame(video_tracker.get_annotated_image(active_trackers_only=False))

class MaskedBackgroundFrameDumper(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='masked_background_frames.mp4')

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()
        self.writer.write_original_frame(video_tracker.get_image('masked_background'))
