import cv2
from threading import Thread

class EventPublisher():

    def __init__(self):
        self._listeners = []

    def listen(self, listener):
        self._listeners.append(listener)

    @property
    def listeners(self):
        return self._listeners

    def publish_process_frame(self, video_tracker):
        listener_threads = []
        for listener in self.listeners:
            listener_thread = Thread(target=self._publish_process_frame_task, args=(listener, video_tracker))
            listener_thread.start()
            listener_threads.append(listener_thread)

        for listener_thread in listener_threads:
            listener_thread.join()

    def publish_finalise(self, total_trackers_started, total_trackers_finished):
        listener_threads = []
        for listener in self.listeners:
            listener_thread = Thread(target=self._publish_finalise_task, args=(listener, total_trackers_started, total_trackers_finished))
            listener_thread.start()
            listener_threads.append(listener_thread)

        for listener_thread in listener_threads:
            listener_thread.join()

    def _publish_process_frame_task(self, listener, video_tracker):
        listener.trackers_updated_callback(video_tracker)

    def _publish_finalise_task(self, listener, total_trackers_started, total_trackers_finished):
        listener.finish(total_trackers_started, total_trackers_finished)
