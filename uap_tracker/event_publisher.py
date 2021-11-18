import cv2

class EventPublisher():

    def __init__(self):
        self._listeners = []

    def listen(self, listener):
        self._listeners.append(listener)

    @property
    def listeners(self):
        return self._listeners

    def publish_process_frame(self, video_tracker):
        for listener in self.listeners:
            listener.trackers_updated_callback(video_tracker)

    def publish_finalise(self, total_trackers_started, total_trackers_finished):
        for listener in self.listeners:
            listener.finish(total_trackers_started, total_trackers_finished)
