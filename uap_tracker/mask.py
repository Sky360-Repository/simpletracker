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

import cv2
import os
import numpy as np
import uap_tracker.utils as utils

####################################################################################################################################
# Base class for various masking implementations. The idea here is that we have a standardised masking processing interface        #
# that is used by the frame processor. We currently support several types which in turn support both CPU and GPU architectures.    #
# If additonal architectures are to be supported in future, like VPI, then this is where the specialisation implementation will go.#
####################################################################################################################################
class Mask():

    # Static factory select method to determine what masking implementation to use
    @staticmethod
    def Select(settings):

        mask_type = settings['mask_type']
        enable_cuda = settings['enable_cuda']

        if mask_type == 'no_op':
            return Mask.NoOp(settings)

        if mask_type == 'fish_eye':
            if enable_cuda:
                print(f'The Fisheye Mask does not support CUDA, the NoOp Mask has been selected. Select Overlay or Overlay_Inverse and provide a masking image for CUDA support.')
                return Mask.NoOp(settings)
            else:
                return Mask.Fisheye(settings)

        if mask_type == 'overlay':
            if enable_cuda:
                return Mask.OverlayGpu(settings)
            else:
                return Mask.Overlay(settings)

        if mask_type == 'overlay_inverse':
            if enable_cuda:
                return Mask.OverlayInverseGpu(settings)
            else:
                return Mask.OverlayInverse(settings)            

        print(f'The NoOp Mask has been selected, is this correct or is there a config error?')
        return Mask.NoOp(settings)

    @staticmethod
    def NoOp(settings):
        return NoOpMask(settings)

    @staticmethod
    def Fisheye(settings):
        return FisheyeMask(settings)

    @staticmethod
    def Overlay(settings):
        return OverlayMask(settings)

    @staticmethod
    def OverlayInverse(settings):
        return OverlayInverseMask(settings)

    @staticmethod
    def OverlayGpu(settings):
        return OverlayMaskGpu(settings)

    @staticmethod
    def OverlayInverseGpu(settings):
        return OverlayInverseMaskGpu(settings)

    def __init__(self):
        pass
    
    # Method to initialise the mask
    def initialise(self, init_frame):
        pass

    # Method to apply the mask to the frame
    def apply(self, frame):
        pass

#############################################################################################################
# NoOp masking implementations. It's just a passthrough and does not perform any sort of masking operation. #
# Its the fallback option and supports both CPU and GPU architectures.                                      #
#############################################################################################################
class NoOpMask(Mask):

    def __init__(self, settings):
        pass

    def initialise(self, init_frame):
        self.shape = init_frame.shape[:2]
        self.height = self.shape[0]
        self.width = self.shape[1]
        return (self.width, self.height)

    def apply(self, frame, stream=None):
        return frame

##################################################################################################################
# Fisheye masking implementations. It provides a percentage based circle shaped mask and also zooms into (clips) #
# the frame once the mask is applied.                                                                            #
# This mask currently only supports the CPU architectures.                                                       #
##################################################################################################################
class FisheyeMask(Mask):

    def __init__(self, settings):
        self.mask_pct = settings['mask_pct']

    def initialise(self, init_frame):
        self.mask_height = (100 - self.mask_pct) / 100.0
        self.mask_radius = self.mask_height / 2.0
        self.shape = init_frame.shape[:2]
        self.height = self.shape[0]
        self.width = self.shape[1]
        self.new_width = int(self.mask_height * self.width)
        self.new_height = int(self.mask_height * self.height)
        return (self.new_width, self.new_height)

    def apply(self, frame, stream=None):
        mask = np.zeros(self.shape, dtype=np.uint8)
        cv2.circle(mask, (int(self.width / 2), int(self.height / 2)), int(min(self.height, self.width) * self.mask_radius), 255, -1)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        clipped_masked_frame = utils.clip_at_center(
            masked_frame,
            (int(self.width / 2), int(self.height / 2)),
            self.width,
            self.height,
            self.new_width,
            self.new_height)
        return clipped_masked_frame

##################################################################################################################
# Overlay masking implementations. It provides the ability to overlay an image over the frame, black is the mask #
# white will form the display area.                                                                              #
# This mask currently only supports both CPU and GPU architectures. This is the CPU implementation.              #
##################################################################################################################
class OverlayMask(Mask):

    def __init__(self, settings):
        self.overlay_image_path = settings['overlay_image_path']

    def initialise(self, init_frame):
        self.shape = init_frame.shape[:2]
        self.height = self.shape[0]
        self.width = self.shape[1]

        # load the oimage we are going to use as a mask
        self.overlay_image = cv2.imread(self.overlay_image_path, cv2.IMREAD_GRAYSCALE)
        
        # resize this image to fit our input frame
        overlay_h = self.overlay_image.shape[:2][0]
        overlay_w = self.overlay_image.shape[:2][1]
        if overlay_h != self.height or overlay_w != self.width:
            print(f'Resizing mask to fit init frame, for better performance ensure your mask matches the same size as your frame h:{self.height}, w:{self.width}.')
            self.overlay_image = cv2.resize(self.overlay_image, (self.width, self.height))

        return (self.width, self.height)

    def apply(self, frame, stream=None):
        masked_frame = cv2.bitwise_and(frame, frame, mask=self.overlay_image)
        return masked_frame

##################################################################################################################
# Overlay masking implementations. It provides the ability to overlay an image over the frame, black is the mask #
# white will form the display area.                                                                              #
# This mask currently only supports both CPU and GPU architectures. This is the GPU implementation.              #
##################################################################################################################
class OverlayMaskGpu(Mask):

    def __init__(self, settings):
        self.overlay_image_path = settings['overlay_image_path']

    def initialise(self, init_frame):
        self.shape = init_frame.shape[:2]
        self.height = self.shape[0]
        self.width = self.shape[1]

        # load the image we are going to use as a mask
        overlay_image = cv2.imread(self.overlay_image_path, cv2.IMREAD_GRAYSCALE)
        
        # resize this image to fit our input frame
        overlay_h = overlay_image.shape[:2][0]
        overlay_w = overlay_image.shape[:2][1]
        if overlay_h != self.height or overlay_w != self.width:
            print(f'Resizing mask to fit init frame, for better performance ensure your mask matches the same size as your frame h:{self.height}, w:{self.width}.')
            overlay_image = cv2.resize(overlay_image, (self.width, self.height))

        # upload to GPU memory for processing on the GPU
        self.gpu_overlay_image = cv2.cuda_GpuMat()
        self.gpu_overlay_image.upload(overlay_image, stream=None)

        return (self.width, self.height)

    def apply(self, gpu_frame, stream):
        # -- work around --
        gpu_masked_frame = cv2.cuda.bitwise_not(gpu_frame, self.gpu_overlay_image, mask=self.gpu_overlay_image, stream=stream)
        gpu_masked_frame = cv2.cuda.bitwise_not(gpu_masked_frame, gpu_masked_frame, mask=self.gpu_overlay_image, stream=stream)
        # -- end work around --
        #Mike: The below statement should work, but there appears to be a bug in OpenCV at the moment: https://github.com/opencv/opencv/issues/20698 so this is a work around
        #gpu_masked_frame = cv2.cuda.bitwise_and(gpu_frame, gpu_frame, mask=self.gpu_overlay_image, stream=stream)
        return gpu_masked_frame

#######################################################################################################
# Overlay inverse masking implementations. It provides the ability to overlay an image over the frame,#
# black will form the viewing area and the colour white is the mask.                                  #
# This mask currently only supports both CPU and GPU architectures. This is the CPU implementation.   #
#######################################################################################################
class OverlayInverseMask(OverlayMask):

    def __init__(self, settings):
        super().__init__(settings)

    def apply(self, frame, stream=None):
        mask = cv2.bitwise_not(self.overlay_image)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame

#######################################################################################################
# Overlay inverse masking implementations. It provides the ability to overlay an image over the frame,#
# black will form the viewing area and the colour white is the mask.                                  #
# This mask currently only supports both CPU and GPU architectures. This is the GPU implementation.   #
#######################################################################################################
class OverlayInverseMaskGpu(OverlayMaskGpu):

    def __init__(self, settings):
        super().__init__(settings)

    def apply(self, gpu_frame, stream):
        gpu_mask = cv2.cuda.bitwise_not(self.gpu_overlay_image, stream=stream)
        # -- work around --
        gpu_masked_frame = cv2.cuda.bitwise_not(gpu_frame, gpu_mask, mask=gpu_mask, stream=stream)
        gpu_masked_frame = cv2.cuda.bitwise_not(gpu_masked_frame, gpu_masked_frame, mask=gpu_mask, stream=stream)
        # -- end work around --
        #Mike: The below statement should work, but there appears to be a bug in OpenCV at the moment: https://github.com/opencv/opencv/issues/20698 so this is a work around
        #gpu_masked_frame = cv2.cuda.bitwise_and(gpu_frame, gpu_frame, mask=gpu_mask, stream=stream)
        return gpu_masked_frame
