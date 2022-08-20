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
from config import settings

class AppSettings():

    @staticmethod
    def Get(settings, detection_mode):

        app_settings = {}

        app_settings['detection_mode'] = detection_mode

        # Video Tracker section
        app_settings['enable_stopwatch'] = settings.VideoTracker.get('enable_stopwatch', False)
        app_settings['enable_cuda'] = settings.VideoTracker.get('enable_cuda', False)
        app_settings['detection_sensitivity'] = settings.VideoTracker.get('sensitivity', 2)
        app_settings['noise_reduction'] = settings.VideoTracker.get('noise_reduction', False)
        app_settings['resize_frame'] = settings.VideoTracker.get('resize_frame', False)
        app_settings['resize_dimension'] = settings.VideoTracker.get('resize_dimension', 1024)
        app_settings['blur_radius'] = settings.VideoTracker.get('blur_radius', 3)
        app_settings['calculate_optical_flow'] = settings.VideoTracker.get('calculate_optical_flow', False)
        app_settings['max_active_trackers'] = settings.VideoTracker.get('max_active_trackers', 10)
        
        app_settings['tracker_type'] = 'CSRT'
        app_settings['tracker_stationary_check_threshold'] = settings.VideoTracker.get('stationary_check_threshold', 5)
        app_settings['tracker_orphaned_check_threshold'] = settings.VideoTracker.get('orphaned_check_threshold', 20)        

        # Mask section
        app_settings['mask_type'] = settings.Mask.get('type', 'fish_eye')
        app_settings['mask_pct'] = settings.Mask.get('mask_pct', 10)

        return app_settings
