import os
import cv2
import Utils as u

class TrackerListener():

    def __init__(self, video, full_path, file_name, output_dir):
        self.video = video
        self.full_path = full_path
        self.file_name = file_name
        self.output_dir = output_dir
        self.recording = False

        self.movement_dir = self._create_output_dir('movement/')
        self.no_movement_dir = self._create_output_dir('no_movement/')
        self.transcoded_dir = self._create_output_dir('transcoded/')

        self.writer_filename = f"{self.transcoded_dir}{self.file_name}.mp4"
        self.writer = None

        print(f"TrackerListener processing {full_path}")

    def _create_output_dir(self, dir_ext):
        dir_to_create = self.output_dir + dir_ext
        if not os.path.isdir(dir_to_create):
            os.mkdir(dir_to_create)
        return dir_to_create

    def trackers_updated_callback(self, frame, frame_id, alive_trackers, fps):
        if self.recording or len(alive_trackers) > 0:
            self.recording = True
            if self.writer is None:
                self._init_writer()
            for tracker in alive_trackers:
                tracker.add_bbox_to_image(frame, (0, 255, 0))
            self.writer.write(frame)

    def finish(self, total_trackers_started, total_trackers_finished):
        if self.writer:
            self.writer.release()
            print(f"TrackerListener detected movement in {self.full_path} transcoded to {self.writer_filename}")
            os.rename(self.full_path, self.movement_dir + os.path.basename(self.full_path))
        else:
            print(f"TrackerListener no movement detected in {self.full_path}")
            os.rename(self.full_path, self.no_movement_dir + os.path.basename(self.full_path))

    def _init_writer(self):
        source_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = u.get_writer(self.writer_filename, source_width, source_height)