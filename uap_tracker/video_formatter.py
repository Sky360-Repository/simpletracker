import os
import cv2
from datetime import datetime, timedelta

from tracker_listener_stf import STFWriter


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

    def trackers_updated_callback(self, video_tracker):

        if not self.writer:
            self._start_video()

        self.writer.write_original_frame(video_tracker.get_image('original'))

    def finish(self, total_trackers_started, total_trackers_finished):
        self._finish_video()
        print(f"Finished processing {self.file_name}")