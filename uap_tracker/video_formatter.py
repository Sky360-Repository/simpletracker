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
from datetime import datetime

from uap_tracker.tracker_listener_stf import STFWriter


class VideoFormatter():

    def __init__(self, source, file_name, output_dir):
        self.file_name = file_name
        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.video_source = source
        self.writer = None
        self.video_start_time = None

    def _start_video(self):
        self.video_start_time = datetime.now()
        self.writer = self._create_stf_writer()
        print(f"VideoFormatter opening video writer {self.writer.video_id}")

    def _finish_video(self):
        print(f"VideoFormatter closing video writer {self.writer.video_id}")
        self.writer.close(min_annotations=-1)
        self.writer = None

    def _source_video_width_height(self):
        source_width = int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return source_width, source_height

    def _create_stf_writer(self):
        width, height = self._source_video_width_height()
        return STFWriter(self.output_dir, self.file_name, width, height)

    def _get_frame_to_write(self, video_tracker):
        return video_tracker.get_image(video_tracker.FRAME_TYPE_ORIGINAL)

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()

        frame = self._get_frame_to_write(video_tracker)
        if frame is not None:
            self.writer.write_original_frame(frame)

    def finish(self, total_trackers_started, total_trackers_finished):
        self._finish_video()
        print(f"Finished processing {self.file_name}")
