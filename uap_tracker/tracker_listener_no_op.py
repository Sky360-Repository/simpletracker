

class TrackerListenerNoOp():

    def __init__(self, video=None, full_path=None, file_name=None, output_dir=None):
        pass

    def initialise(self, sensitivity, blur, normalise_video, tracker_type, background_subtractor_type, source_width, source_height):
        print(f"No Op Listener --> initialise")

    def trackers_updated_callback(self, frame, frame_gray, frame_masked_background, frame_id, alive_trackers, fps):
        print(f"No Op Listener --> trackers_updated_callback")

    def finish(self, total_trackers_started, total_trackers_finished):
        print(f"No Op Listener --> finish")
