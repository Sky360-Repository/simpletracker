import os
import cv2
import Utils as u


#
# Listener to create supervise.ly video format output
#
class TrackerListenerSly():

    def __init__(self, video, full_path, file_name, output_dir):
        self.video = video
        self.full_path = full_path
        self.file_name = file_name
        self.output_dir = output_dir
        self.recording = False
        self._create_output_dir('/sly')
        self.dataset_dir = self._create_output_dir('/sly/ds0/')
        self.ann_dir = self._create_output_dir('/sly/ds0/ann/')
        self.video_dir = self._create_output_dir('/sly/ds0/video/')
        self.processed_dir = self._create_output_dir('/processed/')

        self.video_id=0
        self.writer = None
        self.video_filename=None

        print(f"TrackerListenerSly processing {full_path}")

    def _get_writer_filename_and_increment_id(self):
        filename=f"{self.video_dir}/{self.file_name}_{self.video_id:06}.mp4"
        return filename


    def _create_output_dir(self, dir_ext):
        dir_to_create = self.output_dir + dir_ext
        if not os.path.isdir(dir_to_create):
            os.mkdir(dir_to_create)
        return dir_to_create

    def trackers_updated_callback(self, frame, alive_trackers, fps):
        if len(alive_trackers) > 0:
            if self.writer is None:
                self._init_writer()
            for tracker in alive_trackers:
                #tracker.add_bbox_to_image(frame, (0, 255, 0))
                print(f"{tracker.id}, {tracker.get_bbox()}")
            self.writer.write(frame)
        else:
            if self.writer:
                # No more live trackers, so close out this video
                self._close_writer()



    def finish(self, total_trackers_started, total_trackers_finished):
        if self.writer:
            self.writer.release()
        os.rename(self.full_path, self.processed_dir + os.path.basename(self.full_path))


    def _init_writer(self):
        self.video_filename = self._get_writer_filename_and_increment_id()
        source_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = u.get_writer(self.video_filename, source_width, source_height)

    def _close_writer(self):
        self.writer.release()
        self.writer=None
