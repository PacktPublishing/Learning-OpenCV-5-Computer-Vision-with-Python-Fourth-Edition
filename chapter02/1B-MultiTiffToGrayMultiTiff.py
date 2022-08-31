import cv2
import sys

success, grayImages = cv2.imreadmulti(
    'MyMultiPics.tiff', flags=cv2.IMREAD_GRAYSCALE)
if not success:
    print('Failed to read images from file')
    sys.exit(1)
print('Number of images:', len(grayImages))

success = cv2.imwritemulti('MyMultiPicsGray.tiff', grayImages)
if not success:
    print('Failed to write images to file')
    sys.exit(1)
