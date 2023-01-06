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


def showResult(Rt, bgrFrame):

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

    # Resize the frame for display.
    resizedFrame = cv2.resize(bgrFrame, (960, 540))

    # Print the odometry result on the resized frame.
    textColor = (192, 64, 192)
    cv2.putText(resizedFrame, 'yaw (deg.): %f' % eulerDegrees[0],
                (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(resizedFrame, 'pitch (deg.): %f' % eulerDegrees[1],
                (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(resizedFrame, 'roll (deg.): %f' % eulerDegrees[2],
                (5, 85), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(resizedFrame, 'x: %f' % position[0],
                (5, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(resizedFrame, 'y: %f' % position[1],
                (5, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)
    cv2.putText(resizedFrame, 'z: %f' % position[2],
                (5, 190), cv2.FONT_HERSHEY_SIMPLEX, 1,
                textColor, 2)

    # Show the resized frame.
    cv2.imshow("Result", resizedFrame)


w = 1280
h = 720
diagonal_fov_degrees = 70.0

odometryType = cv2.ODOMETRY_TYPE_RGB
odometrySettings = cv2.OdometrySettings()
odometrySettings.setCameraMatrix(
    createCameraMatrix(w, h, diagonal_fov_degrees))
odometryAlgoType = cv2.ODOMETRY_ALGO_TYPE_COMMON
odometry = cv2.Odometry(odometryType, odometrySettings, odometryAlgoType)

videoCapture = cv2.VideoCapture(0)
videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

lastOdometryFrame = None

Rt = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

while cv2.waitKey(1) == -1:

    success, bgrFrame = videoCapture.read()
    if not success:
        continue

    odometryFrame = cv2.OdometryFrame(None, bgrFrame)
    if lastOdometryFrame is not None:
        odometry.prepareFrames(lastOdometryFrame, odometryFrame)
        success, RtTemp = odometry.compute(
            lastOdometryFrame, odometryFrame)
        if success:
            cv2.gemm(Rt, RtTemp, 1.0, None, 0.0, Rt)
    showResult(Rt, bgrFrame)
    lastOdometryFrame = odometryFrame
