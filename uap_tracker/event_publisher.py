import cv2

class EventPublisher():

    def __init__(self):
        self._listeners = []

    def listen(self, listener):
        self._listeners.append(listener)

    @property
    def listeners(self):
        return self._listeners

    def publish_initialise(self, sensitivity, blur, normalise_video, tracker_type, background_subtractor_type, source_width, source_height):
        for listener in self.listeners:
            listener.initialise(sensitivity, blur, normalise_video, tracker_type, background_subtractor_type, source_width, source_height)

    def publish_process_frame(self, frame, frame_gray, frame_masked_background, optical_flow_frame, frame_id, alive_trackers, fps):
        for listener in self.listeners:
            print(listener)
            listener.trackers_updated_callback(
                frame, frame_gray, frame_masked_background, optical_flow_frame, frame_id, alive_trackers, fps
            )

    def publish_finalise(self, total_trackers_started, total_trackers_finished):
        for listener in self.listeners:
            listener.finish(total_trackers_started, total_trackers_finished)
