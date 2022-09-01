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

import sys
import os
from config import settings

##################################################################################################
# This class provides a central area for populating and validating the application configuration #
##################################################################################################
class AppSettings():

    # Method to populate a configuration dictionary for use throughout the simple tracker application
    @staticmethod
    def Get(settings):

        app_settings = {}

        app_settings['controller'] = settings.get('controller', None)

        app_settings['camera_mode'] = settings.Camera.get('camera_mode', 'rtsp')
        app_settings['camera_uri'] = settings.Camera.get('camera_uri', None)
        app_settings['camera_iteration_interval'] = settings.Camera.get('camera_iteration_interval', 10)

        #Visualisers
        app_settings['font_size'] = settings.Visualizer.get('font_size', 0.75)
        app_settings['font_thickness'] = settings.Visualizer.get('font_thickness', 2)

        # Video Tracker section
        app_settings['enable_stopwatch'] = settings.VideoTracker.get('enable_stopwatch', False)
        app_settings['enable_cuda'] = settings.VideoTracker.get('enable_cuda', False)
        app_settings['detection_mode'] = settings.VideoTracker.get('detection_mode', None)
        app_settings['detection_sensitivity'] = settings.VideoTracker.get('sensitivity', 2)
        app_settings['noise_reduction'] = settings.VideoTracker.get('noise_reduction', False)
        app_settings['resize_frame'] = settings.VideoTracker.get('resize_frame', False)
        app_settings['resize_dimension'] = settings.VideoTracker.get('resize_dimension', 1024)
        app_settings['blur_radius'] = settings.VideoTracker.get('blur_radius', 3)
        app_settings['calculate_optical_flow'] = settings.VideoTracker.get('calculate_optical_flow', False)
        app_settings['max_active_trackers'] = settings.VideoTracker.get('max_active_trackers', 10)
        app_settings['tracker_type'] = settings.VideoTracker.get('tracker_type', 'CSRT')

        # Tracker section
        app_settings['enable_track_validation'] = settings.VideoTracker.get('enable_track_validation', True)
        app_settings['stationary_track_threshold'] = settings.VideoTracker.get('stationary_track_threshold', 5)
        app_settings['orphaned_track_threshold'] = settings.VideoTracker.get('orphaned_track_threshold', 20)        

        # BBox section
        app_settings['bbox_fixed_size'] = settings.VideoTracker.get('bbox_fixed_size', False)
        app_settings['bbox_size'] = settings.VideoTracker.get('bbox_size', 64)

        # Track Plotting section
        app_settings['track_plotting_enabled'] = settings.VideoTracker.get('track_plotting_enabled', False)
        app_settings['track_plotting_type'] = settings.VideoTracker.get('track_plotting_type', 'line')

        # Track Prediction section
        app_settings['track_prediction_enabled'] = settings.VideoTracker.get('track_prediction_enabled', False)

        # Mask section
        app_settings['mask_type'] = settings.Mask.get('type', 'fish_eye')
        app_settings['mask_pct'] = settings.Mask.get('mask_pct', 10)
        app_settings['overlay_image_path'] = settings.Mask.get('overlay_image_path', None)

        # Dense optical flow
        app_settings['dense_optical_flow_height'] = 480
        app_settings['dense_optical_flow_width'] = 480

        return app_settings

    # Method used to validate configuration dictionary for use throughout the simple tracker application
    # If there is a specific combination of configuration that can't be used or that has to be used
    # this is where the validation of that combination should happen
    @staticmethod
    def Validate(app_settings):

        controller = app_settings['controller']
        if controller == 'camera':
            if app_settings['camera_uri'] == None:
                print(f"Controller is set to Camera but camera_uri is None")
                sys.exit(2)

        detection_mode = app_settings['detection_mode']
        detection_modes = ['background_subtraction', 'optical_flow', 'none']
        if not detection_mode:
            print(
                f"Please set detection_mode in the config or use the SKY360_DETECTION_MODE env var: {detection_modes}")
            sys.exit(1)
        else:
            print(f"Detection Mode: {detection_mode}")

        detection_sensitivity = app_settings['detection_sensitivity']
        if detection_sensitivity < 1 or detection_sensitivity > 3:
            print(
                f"Unknown sensitivity option ({detection_sensitivity}). 1, 2 and 3 is supported not {detection_sensitivity}.")
            sys.exit(1)

        mask_type = app_settings['mask_type']
        if mask_type == 'fish_eye':
            mask_pct = app_settings['mask_pct']
            if mask_pct < 1:
                app_settings['mask_type'] = 'no_op'
                print(f"You have selected a Fish Eye mask type but the mask_pct = {mask_pct}, a no op mask will be used.")

        if mask_type == 'overlay' or mask_type == 'overlay_inverse':
            overlay_image_path = app_settings['overlay_image_path']
            if overlay_image_path == None or os.path.exists(overlay_image_path) == False:
                app_settings['mask_type'] = 'no_op'                
                print(f"You have selected an {mask_type} mask type but the masking image '{overlay_image_path}' can't be found, a no_op mask will be used.")

        track_plotting_type = app_settings['track_plotting_type']
        if not track_plotting_type == 'line' or track_plotting_type == 'dot':
            print(f"You have selected an unsupported track plotting type {track_plotting_type}, it will be reset to line.")
            app_settings['track_plotting_type'] = 'line'
