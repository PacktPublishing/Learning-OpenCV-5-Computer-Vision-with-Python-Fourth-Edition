import cv2
import depthai as dai
import numpy as np


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

####


# Define the nodes in the RGB chain.
rgb = pipeline.create(dai.node.ColorCamera)
rgbOut = pipeline.create(dai.node.XLinkOut)
rgbOut.setStreamName("rgb")

# Configure the RGB camera.
rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb.setResolution(
    dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb.setVideoSize(1920, 1080)


# Link the nodes in the mono -> stereo -> output chain.
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(disparityOut.input)

# Link the nodes in the RGB -> output chain.
rgb.video.link(rgbOut.input)


# Connect to the device and start the pipeline.
with dai.Device(pipeline) as device:

    # Get the output queues.
    disparityQ = device.getOutputQueue(
        name="disparity", maxSize=4, blocking=False)
    rgbQ = device.getOutputQueue(
        name="rgb", maxSize=4, blocking=False)

    while cv2.waitKey(1) == -1:

        # Grab the next disparity frame, if any is ready.
        disparityGrab = disparityQ.tryGet()

        # Grab the next RGB frame, if any is ready.
        rgbGrab = rgbQ.tryGet()

        if disparityGrab is not None:
            disparityFrame = disparityGrab.getFrame()
            # Normalize the disparity map for better visualization.
            disparityFrame = (disparityFrame * \
                (255.0 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)
            cv2.imshow("disparity", disparityFrame)

        if rgbGrab is not None:
            # Get the frame and convert it from its native NV12 encoding
            # to OpenCV's BGR format.
            bgrFrame = rgbGrab.getCvFrame()
            cv2.imshow("RGB", bgrFrame)
