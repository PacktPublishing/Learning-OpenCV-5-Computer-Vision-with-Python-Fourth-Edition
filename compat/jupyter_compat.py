import os

import cv2
import numpy
import PIL.Image

from IPython import display
from urllib.request import urlopen


def cv2_imshow(winname, mat):
    mat = mat.clip(0, 255).astype('uint8')
    if mat.ndim == 3:
        if mat.shape[2] == 4:
            mat = cv2.cvtColor(mat, cv2.COLOR_BGRA2RGBA)
        else:
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    display.display(PIL.Image.fromarray(mat))

cv2.imshow = cv2_imshow


def cv2_waitKey(delay=0):
    return -1

cv2.waitKey = cv2_waitKey


def cv2_imread(filename, flags=cv2.IMREAD_COLOR):
    if os.path.exists(filename):
        image = cv2._imread(filename, flags)
    else:
        url = f'https://github.com/PacktPublishing/Learning-OpenCV-5-Computer-Vision-with-Python-Fourth-Edition/raw/main/*/{filename}'
        resp = urlopen(url)
        image = numpy.asarray(bytearray(resp.read()), dtype='uint8')
        image = cv2.imdecode(image, flags)
    return image

cv2._imread = cv2.imread
cv2.imread = cv2_imread
