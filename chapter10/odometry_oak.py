#!/usr/bin/env python

import math

import cv2
import depthai as dai
import numpy as np


def createCameraMatrix(w, h, diagonal_fov_degrees):
    diagonal_image_size = (w ** 2.0 + h ** 2.0) ** 0.5
    diagonal_fov_radians = \
        diagonal_fov_degrees * math.pi / 180.0
    focal_length = 0.5 * diagonal_image_size / math.tan(
        0.5 * diagonal_fov_radians)
    return np.array(
        [[focal_length, 0.0, 0.5 * w],
         [0.0, focal_length, 0.5 * h],
         [0.0, 0.0, 1.0]], np.float32)


def showResult(Rt, bgrFrame, disparityFrame):

    m00 = Rt[0, 0]
    m02 = Rt[0, 2]
    m10 = Rt[1, 0]
    m11 = Rt[1, 1]
    m12 = Rt[1, 2]
    m20 = Rt[2, 0]
    m22 = Rt[2, 2]

    # Convert to Euler angles using the yaw-pitch-roll
    # Tait-Bryan convention.
    if m10 > 0.998:
        # The rotation is near the "vertical climb" singularity.
        pitch = 0.5 * math.pi
        yaw = math.atan2(m02, m22)
        roll = 0.0
    elif m10 < -0.998:
        # The rotation is near the "nose dive" singularity.
        pitch = -0.5 * math.pi
        yaw = math.atan2(m02, m22)
        roll = 0.0
    else:
        pitch = math.asin(m10)
        yaw = math.atan2(-m20, m00)
        roll = math.atan2(-m12, m11)

    eulerDegrees = np.rad2deg([pitch, yaw, roll])
    position = Rt[:,3][:3]

    # Normalize the disparity map for better visualization.
    disparityFrame = (disparityFrame * \
        (255.0 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

    # Apply false colorization to the disparity map.
    disparityFrame = cv2.applyColorMap(
        disparityFrame, cv2.COLORMAP_HOT)

    # Blend the BGR frame and the colorized disparity map.
    blendedFrame = cv2.addWeighted(
        bgrFrame, 0.4, disparityFrame, 0.6, 0.0)

    # Resize the blended frame for display.
    blendedFrame = cv2.resize(blendedFrame, (960, 540))

    # Print the odometry result on the blended frame.
    textColor = (192, 64, 192)
    cv2.putText(blendedFrame, 'yaw (deg.): %f' % eulerDegrees[0],
                (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(blendedFrame, 'pitch (deg.): %f' % eulerDegrees[1],
                (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(blendedFrame, 'roll (deg.): %f' % eulerDegrees[2],
                (5, 85), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(blendedFrame, 'x: %f' % position[0],
                (5, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(blendedFrame, 'y: %f' % position[1],
                (5, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(blendedFrame, 'z: %f' % position[2],
                (5, 190), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)

    # Show the blended frame.
    cv2.imshow("Result", blendedFrame)


odometryType = cv2.ODOMETRY_TYPE_RGB_DEPTH
odometrySettings = cv2.OdometrySettings()
odometrySettings.setCameraMatrix(
    createCameraMatrix(1920, 1080, 120.0))
odometryAlgoType = cv2.ODOMETRY_ALGO_TYPE_COMMON
odometry = cv2.Odometry(odometryType, odometrySettings, odometryAlgoType)


# Create the pipeline.
pipeline = dai.Pipeline()


# Define the nodes in the mono -> stereo -> output chain.
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
disparityOut = pipeline.create(dai.node.XLinkOut)
disparityOut.setStreamName("disparity")

# Configure the mono cameras.
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
monoLeft.setResolution(monoResolution)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(monoResolution)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

####
# Configure the depth node for disparity output.

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

# True -> nearer minimum depth (max disparity = 190)
# False -> farther minimum depth (max disparity = 95)
depth.setExtendedDisparity(False)

# True -> better accuracy at far depth, but
#         incompatible with setExtendedDisparity(True)
depth.setSubpixel(False)

# Align the depth output to the RGB output.
depth.setDepthAlign(dai.CameraBoardSocket.RGB)

####

# Link the nodes in the mono -> stereo -> output chain.
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(disparityOut.input)


# Define the nodes in the RGB chain.
rgb = pipeline.create(dai.node.ColorCamera)
rgbOut = pipeline.create(dai.node.XLinkOut)
rgbOut.setStreamName("rgb")

# Configure the RGB camera.
rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb.setResolution(
    dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb.setVideoSize(1920, 1080)

# Link the nodes in the RGB -> output chain.
rgb.video.link(rgbOut.input)


# Connect to the device and start the pipeline.
with dai.Device(pipeline) as device:

    # Get the output queues.
    disparityQ = device.getOutputQueue(
        name="disparity", maxSize=4, blocking=False)
    rgbQ = device.getOutputQueue(
        name="rgb", maxSize=4, blocking=False)

    lastBGRFrame = None
    lastOdometryFrame = None

    Rt = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    while cv2.waitKey(1) == -1:

        # Grab the next disparity frame, if any is ready.
        disparityGrab = disparityQ.tryGet()

        # Grab the next RGB frame, if any is ready.
        rgbGrab = rgbQ.tryGet()

        if rgbGrab is not None:
            # Get the frame and convert it from its native NV12 encoding
            # to OpenCV's BGR format.
            lastBGRFrame = rgbGrab.getCvFrame()

        if disparityGrab is not None and lastBGRFrame is not None:

            disparityFrame = disparityGrab.getFrame()

            # Invalid pixels have the value 0 in the disparity map.
            mask = np.where(
                disparityFrame == 0, 0, 255).astype(np.uint8)

            odometryFrame = cv2.OdometryFrame(
                disparityFrame, lastBGRFrame, mask)

            if lastOdometryFrame is not None:
                odometry.prepareFrames(lastOdometryFrame, odometryFrame)
                success, RtTemp = odometry.compute(
                    lastOdometryFrame, odometryFrame)
                if success:
                    cv2.gemm(Rt, RtTemp, 1.0, None, 0.0, Rt)
                    showResult(Rt, lastBGRFrame, disparityFrame)
            lastOdometryFrame = odometryFrame
