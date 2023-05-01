#!/usr/bin/env python

import cv2
import depthai as dai
import numpy as np


w = 1920
h = 1080

# Skip some frames at the start of the recording
# to ensure that the autoexposure has time to adjust.
numFramesToSkip = 10


# Create the pipeline.
pipeline = dai.Pipeline()


# Define the nodes in the mono -> stereo -> output chain.
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName('depth')

# Configure the mono cameras.
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
monoLeft.setResolution(monoResolution)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(monoResolution)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

####
# Configure the depth node for depth output.

depth.setDefaultProfilePreset(
    dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# Median filter options:
#     MEDIAN_OFF
#     KERNEL_3x3
#     KERNEL_5x5
#     KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

# True -> better occlusion handling
depth.setLeftRightCheck(True)

# Align the depth output to the RGB output.
depth.setDepthAlign(dai.CameraBoardSocket.RGB)

####

# Link the nodes in the mono -> stereo -> output chain.
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.depth.link(depthOut.input)


# Define the nodes in the RGB chain.
rgb = pipeline.create(dai.node.ColorCamera)
rgbOut = pipeline.create(dai.node.XLinkOut)
rgbOut.setStreamName('rgb')

# Configure the RGB camera.
rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb.setResolution(
    dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb.setVideoSize(w, h)

# Link the nodes in the RGB -> output chain.
rgb.video.link(rgbOut.input)


# Connect to the device and start the pipeline.
with dai.Device(pipeline) as device:

    # Get the output queues.
    depthQ = device.getOutputQueue(
        name='depth', maxSize=4, blocking=False)
    rgbQ = device.getOutputQueue(
        name='rgb', maxSize=4, blocking=False)

    lastBGRFrame = None
    allBGRFrames = []
    allDepthFrames = []

    while cv2.waitKey(1) == -1:

        # Grab the next depth frame, if any is ready.
        depthGrab = depthQ.tryGet()

        # Grab the next RGB frame, if any is ready.
        rgbGrab = rgbQ.tryGet()

        if rgbGrab is not None:
            # Get the frame and convert it from its native NV12 encoding
            # to OpenCV's BGR format.
            lastBGRFrame = rgbGrab.getCvFrame()

            cv2.imshow('RGB', lastBGRFrame)

        if depthGrab is not None and lastBGRFrame is not None:

            depthFrame = depthGrab.getFrame()

            allBGRFrames.append(lastBGRFrame)
            allDepthFrames.append(depthFrame)

    np.savez_compressed(
        '../videos/oak_bgr_and_depth.npz',
        bgr=allBGRFrames[numFramesToSkip:],
        depth=allDepthFrames[numFramesToSkip:])
