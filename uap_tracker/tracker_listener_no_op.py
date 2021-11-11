

class TrackerListenerNoOp():

    def __init__(self):
        pass

    def initialise(self, sensitivity, blur, normalise_video, tracker_type, background_subtractor_type, source_width, source_height):
        print(f"TrackerListenerNoOp initialise")

    def trackers_updated_callback(self, frame, frame_gray, frame_masked_background, frame_id, alive_trackers, fps):
        print(f"TrackerListenerNoOp trackers_updated_callback")

    def finish(self, total_trackers_started, total_trackers_finished):
        print(f"TrackerListenerNoOp finish")
