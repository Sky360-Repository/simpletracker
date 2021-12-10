import os
import cv2
from datetime import datetime, timedelta
import uap_tracker.utils as utils

class FrameDumper():

    def __init__(self,
                 output_dir,
                 video_file_root_name,
                 source_width,
                 source_height,
                 video_name):

        self.writer = None

        self.video_dir = None
        self.video_filename = None

        self.source_width = source_width
        self.source_height = source_height

        self.video_name = video_name

        self.final_video_dir = output_dir

        if not os.path.isdir(self.final_video_dir):
            os.mkdir(self.final_video_dir)

        self.video_filename = os.path.join(self.final_video_dir, video_name)

    def _close_video_writers(self):
        self.writer.release()
        self.writer = None

    def write_frame(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]
        if not self.writer:
            self.writer = utils.get_writer(
                self.video_filename, width, height)

        self.writer.write(frame)

    def close(self):
        if self.writer:
            self._close_video_writers()

class DumpFormatter():

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
        self.writer = self._create_dumper()
        print(f"Dumper opening writer {self.writer.video_name}")

    def _finish_video(self):
        print(f"Dumper closing writer {self.writer.video_name}")
        self.writer.close()
        self.writer = None

    def _source_video_width_height(self):
        source_width = int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return source_width, source_height

    def _create_dumper(self):
        pass

    def trackers_updated_callback(self, video_tracker):
        pass

    def finish(self, total_trackers_started, total_trackers_finished):
        self._finish_video()

class OriginalFrameDumper(DumpFormatter):

    def _create_dumper(self):
        width, height = self._source_video_width_height()
        return FrameDumper(self.output_dir, self.file_name, width, height, video_name='original_frames_dump.mp4')

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()

        frame = video_tracker.get_image('original')
        if frame is not None:
            self.writer.write_frame(frame)
            #cv2.imshow("OriginalFrameDumper", frame)

class GreyFrameDumper(DumpFormatter):

    def _create_dumper(self):
        width, height = self._source_video_width_height()
        return FrameDumper(self.output_dir, self.file_name, width, height, video_name='grey_frames_dump.mp4')

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()

        grey_frame = video_tracker.get_image('grey')
        if grey_frame is not None:
            self.writer.write_frame(cv2.cvtColor(grey_frame, cv2.COLOR_GRAY2BGR))
            #cv2.imshow("GreyFrameDumper", grey_frame)

class OpticalFlowFrameDumper(DumpFormatter):

    def _create_dumper(self):
        width, height = self._source_video_width_height()
        return FrameDumper(self.output_dir, self.file_name, width, height, video_name='optical_flow_frames_dump.mp4')

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()

        optical_flow_frame = video_tracker.get_image('optical_flow')
        if optical_flow_frame is not None:
            self.writer.write_frame(optical_flow_frame)
            #cv2.imshow("OpticalFlowFrameDumper", optical_flow_frame)

class AnnotatedFrameDumper(DumpFormatter):

    def _create_dumper(self):
        width, height = self._source_video_width_height()
        return FrameDumper(self.output_dir, self.file_name, width, height, video_name='annotated_frames_dump.mp4')

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()

        annotated_frame = video_tracker.get_annotated_image(active_trackers_only=False)
        if annotated_frame is not None:
            self.writer.write_frame(annotated_frame)
            #cv2.imshow("AnnotatedFrameDumper", annotated_frame)

class MaskedBackgroundFrameDumper(DumpFormatter):

    def _create_dumper(self):
        width, height = self._source_video_width_height()
        return FrameDumper(self.output_dir, self.file_name, width, height, video_name='masked_background_frames_dump.mp4')

    def trackers_updated_callback(self, video_tracker):
        if not self.writer:
            self._start_video()

        masked_background_frame = video_tracker.get_image('masked_background')
        if masked_background_frame is not None:
            self.writer.write_frame(cv2.cvtColor(masked_background_frame, cv2.COLOR_GRAY2BGR))
            #cv2.imshow("MaskedBackgroundFrameDumper", masked_background_frame)
