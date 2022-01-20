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

import cv2
from uap_tracker.stf_writer import STFWriter
from uap_tracker.video_formatter import VideoFormatter

class OriginalFrameVideoWriter(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='original_frames.mp4',
                           movement_alpha=False, annotate=False)

    def _get_frame_to_write(self, video_tracker):
        return video_tracker.get_image(video_tracker.FRAME_TYPE_ORIGINAL)

class GreyFrameVideoWriter(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='grey_frames.mp4',
                           movement_alpha=False, annotate=False)

    def _get_frame_to_write(self, video_tracker):
        grey_frame = video_tracker.get_image(video_tracker.FRAME_TYPE_GREY);
        if grey_frame is not None:
            grey_frame = cv2.cvtColor(grey_frame, cv2.COLOR_GRAY2BGR)
        return grey_frame

class OpticalFlowFrameVideoWriter(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='optical_flow_frames.mp4',
                           movement_alpha=False, annotate=False)

    def _get_frame_to_write(self, video_tracker):
        return video_tracker.get_image(video_tracker.FRAME_TYPE_OPTICAL_FLOW)

class AnnotatedFrameVideoWriter(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='annotated_frames.mp4',
                           movement_alpha=False, annotate=False)

    def _get_frame_to_write(self, video_tracker):
        return video_tracker.get_annotated_image(active_trackers_only=False)

class MaskedBackgroundFrameVideoWriter(VideoFormatter):

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height, video_name='masked_background_frames.mp4',
                           movement_alpha=False, annotate=False)

    def _get_frame_to_write(self, video_tracker):
        masked_background_frame = video_tracker.get_image(video_tracker.FRAME_TYPE_MASKED_BACKGROUND);
        if masked_background_frame is not None:
            masked_background_frame = cv2.cvtColor(masked_background_frame, cv2.COLOR_GRAY2BGR)
        return masked_background_frame
