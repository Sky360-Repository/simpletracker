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

from threading import Thread

#####################################################################################################
# This class provides a central area for video tracker to publish processing events to its listners #
#####################################################################################################
class EventPublisher():

    def __init__(self):
        self._listeners = []

    def listen(self, listener):
        self._listeners.append(listener)

    @property
    def listeners(self):
        return self._listeners

    # function to publish the process frame event, we process each listener on its own thread to try and speed things up
    def publish_process_frame(self, video_tracker):
        listener_threads = []
        for listener in self.listeners:
            # self._publish_process_frame_task(listener, video_tracker)
            listener_thread = Thread(target=self._publish_process_frame_task, args=(listener, video_tracker))
            listener_thread.start()
            listener_threads.append(listener_thread)

        for listener_thread in listener_threads:
            listener_thread.join()

    # function to publish the finalise event, we process each listener on its own thread to try and speed things up
    def publish_finalise(self, total_trackers_started, total_trackers_finished):
        listener_threads = []
        for listener in self.listeners:
            # self._publish_finalise_task(listener, total_trackers_started, total_trackers_finished)
            listener_thread = Thread(target=self._publish_finalise_task, args=(listener, total_trackers_started, total_trackers_finished))
            listener_thread.start()
            listener_threads.append(listener_thread)

        for listener_thread in listener_threads:
            listener_thread.join()

    # private function to serve as the thread's entry point
    def _publish_process_frame_task(self, listener, video_tracker):
        listener.trackers_updated_callback(video_tracker)

    # private function to serve as the thread's entry point
    def _publish_finalise_task(self, listener, total_trackers_started, total_trackers_finished):
        listener.finish(total_trackers_started, total_trackers_finished)
