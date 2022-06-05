import cv2

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

def is_inside(i, o):
    ix, iy, iw, ih = i
    ox, oy, ow, oh = o
    return ix > ox and ix + iw < ox + ow and \
        iy > oy and iy + ih < oy + oh

hog = cv2.HOGDescriptor((48, 96), (16, 16), (8, 8), (8, 8), 9)
hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

img = cv2.imread('../images/haying.jpg')

if OPENCV_MAJOR_VERSION >= 5:
    # OpenCV 5 or a later version is being used.
    found_rects, found_weights = hog.detectMultiScale(
        img, winStride=(8, 8), scale=1.04, groupThreshold=6.0)
else:
    # OpenCV 4 or an earlier version is being used.
    # The groupThreshold parameter used to be named finalThreshold.
    found_rects, found_weights = hog.detectMultiScale(
        img, winStride=(8, 8), scale=1.04, finalThreshold=6.0)

found_rects_filtered = []
found_weights_filtered = []
for ri, r in enumerate(found_rects):
    for qi, q in enumerate(found_rects):
        if ri != qi and is_inside(r, q):
            break
    else:
        found_rects_filtered.append(r)
        found_weights_filtered.append(found_weights[ri])

for ri, r in enumerate(found_rects_filtered):
    x, y, w, h = r
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    text = '%.2f' % found_weights_filtered[ri]
    cv2.putText(img, text, (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.imshow('Women in Hayfield Detected', img)
cv2.imwrite('./women_in_hayfield_detected_daimler.png', img)
cv2.waitKey(0)
