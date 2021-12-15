import os
import cv2
from datetime import datetime, timedelta
from uap_tracker.stf_writer import STFWriter
from uap_tracker.video_formatter import VideoFormatter
import uap_tracker.utils as utils

class OriginalFrameVideoWriter(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='original_frames.mp4',
                           movement_alpha=False, annotate=False)

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()

        frame = video_tracker.get_image('original')
        if frame is not None:
            self.writer.write_original_frame(frame)
            #cv2.imshow("OriginalFrameDumper", frame)

class GreyFrameVideoWriter(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='grey_frames.mp4',
                           movement_alpha=False, annotate=False)

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()

        grey_frame = video_tracker.get_image('grey')
        if grey_frame is not None:
            self.writer.write_original_frame(cv2.cvtColor(grey_frame, cv2.COLOR_GRAY2BGR))
            #cv2.imshow("GreyFrameDumper", grey_frame)

class OpticalFlowFrameVideoWriter(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='optical_flow_frames.mp4',
                           movement_alpha=False, annotate=False)

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()

        optical_flow_frame = video_tracker.get_image('optical_flow')
        if optical_flow_frame is not None:
            self.writer.write_original_frame(optical_flow_frame)
            #cv2.imshow("OpticalFlowFrameDumper", optical_flow_frame)

class AnnotatedFrameVideoWriter(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='annotated_frames.mp4',
                           movement_alpha=False, annotate=False)

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()

        annotated_frame = video_tracker.get_annotated_image(active_trackers_only=False)
        if annotated_frame is not None:
            self.writer.write_original_frame(annotated_frame)
            #cv2.imshow("AnnotatedFrameDumper", annotated_frame)

class MaskedBackgroundFrameVideoWriter(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='masked_background_frames.mp4',
                           movement_alpha=False, annotate=False)

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()

        masked_background_frame = video_tracker.get_image('masked_background')
        if masked_background_frame is not None:
            self.writer.write_original_frame(cv2.cvtColor(masked_background_frame, cv2.COLOR_GRAY2BGR))
            #cv2.imshow("MaskedBackgroundFrameDumper", masked_background_frame)
