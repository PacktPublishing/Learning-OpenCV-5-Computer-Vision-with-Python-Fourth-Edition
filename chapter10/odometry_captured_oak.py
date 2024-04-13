#!/usr/bin/env python

import math

import cv2
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


def accumulateResult(Rt, RtStep):

    # Get the 3x3 rotation submatrices
    # and 1x3 translation submatrices.
    rotation = Rt[:3,:3]
    translation = Rt[:,3][:3]
    rotationStep = RtStep[:3,:3]
    translationStep = RtStep[:,3][:3].reshape(3, 1)

    # Update the translation.
    translation += cv2.gemm(
        rotation, translationStep,
        1.0, None, 0.0).squeeze()

    # Update the rotation.
    cv2.gemm(rotation, rotationStep,
        1.0, None, 0.0, rotation)


def showResult(Rt, bgrFrame, depthFrame):

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
    translation = Rt[:,3][:3]

    # Scale the depth map for better visualization.
    depthFrame = np.minimum(
        255.0, (depthFrame / 40.0)).astype(np.uint8)

    # Apply false colorization to the depth map.
    depthFrame = cv2.applyColorMap(
        depthFrame, cv2.COLORMAP_COOL)

    # Blend the BGR frame and the colorized depth map.
    blendedFrame = cv2.addWeighted(
        bgrFrame, 0.7, depthFrame, 0.3, 0.0)

    # Resize the blended frame for display.
    blendedFrame = cv2.resize(blendedFrame, (960, 540))

    # Print the odometry result on the blended frame.
    textColor = (64, 192, 192)
    cv2.putText(blendedFrame, 'yaw (degrees): %f' % eulerDegrees[0],
                (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(blendedFrame, 'pitch (degrees): %f' % eulerDegrees[1],
                (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(blendedFrame, 'roll (degrees): %f' % eulerDegrees[2],
                (5, 85), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(blendedFrame, 'x (meters): %f' % translation[0],
                (5, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(blendedFrame, 'y (meters): %f' % translation[1],
                (5, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(blendedFrame, 'z (meters): %f' % translation[2],
                (5, 190), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)

    # Show the blended frame.
    cv2.imshow('Result', blendedFrame)


w = 1920
h = 1080
diagonal_fov_degrees = 81.0

odometryType = cv2.ODOMETRY_TYPE_RGB_DEPTH
odometrySettings = cv2.OdometrySettings()
odometrySettings.setCameraMatrix(createCameraMatrix(
    w, h, diagonal_fov_degrees))
odometryAlgoType = cv2.ODOMETRY_ALGO_TYPE_FAST
odometry = cv2.Odometry(
    odometryType, odometrySettings, odometryAlgoType)


# Load and process the pre-recorded OAK-D frames.
with open('../videos/oak_bgr_and_depth.npz', 'rb') as f:

    frames = np.load(f)
    bgrFrames = frames['bgr']
    depthFrames = frames['depth']
    numFrames = min(len(bgrFrames), len(depthFrames))

    lastOdometryFrame = None

    Rt = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    i = 0
    while cv2.waitKey(1) == -1 and i < numFrames:

        bgrFrame = bgrFrames[i]
        depthFrame = depthFrames[i]

        # Invalid pixels have the value 0 in the depth map.
        mask = np.where(
            depthFrame == 0, 0, 255).astype(np.uint8)

        odometryFrame = cv2.OdometryFrame(
            depthFrame, bgrFrame, mask)

        if lastOdometryFrame is not None:

            odometry.prepareFrames(lastOdometryFrame, odometryFrame)
            success, RtStep = odometry.compute(
                lastOdometryFrame, odometryFrame)

            if success:
                accumulateResult(Rt, RtStep)

        showResult(Rt, bgrFrame, depthFrame)
        lastOdometryFrame = odometryFrame

        i += 1
