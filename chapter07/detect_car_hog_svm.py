import cv2
import numpy as np
import os

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])
OPENCV_MINOR_VERSION = int(cv2.__version__.split('.')[1])

if not os.path.isdir('CarData'):
    print('CarData folder not found. Please download and unzip '
          'https://github.com/gcr/arc-evaluator/raw/master/CarData.tar.gz '
          'into the same folder as this script.')
    exit(1)

HOG_WINDOW_SIZE = (96, 48)
HOG_WEIGHT_THRESHOLD = 0.45

SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 300

hog = cv2.HOGDescriptor(HOG_WINDOW_SIZE, (16, 16), (8, 8), (8, 8), 9)

def get_pos_and_neg_paths(i):
    pos_path = 'CarData/TrainImages/pos-%d.pgm' % (i+1)
    neg_path = 'CarData/TrainImages/neg-%d.pgm' % (i+1)
    return pos_path, neg_path

def extract_hog_descriptors(img):
    resized = cv2.resize(img, HOG_WINDOW_SIZE, cv2.INTER_CUBIC)
    return hog.compute(resized, (16, 16), (0, 0))

training_data = []
training_labels = []
for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
    pos_descriptors = extract_hog_descriptors(pos_img)
    if pos_descriptors is not None:
        training_data.append(pos_descriptors)
        training_labels.append(1)
    neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
    neg_descriptors = extract_hog_descriptors(neg_img)
    if neg_descriptors is not None:
        training_data.append(neg_descriptors)
        training_labels.append(-1)

svm = cv2.ml.SVM_create()
svm.setDegree(3)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
svm.setTermCriteria(criteria)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setNu(0.5)
svm.setP(0.1)
svm.setC(0.01)
svm.setType(cv2.ml.SVM_EPS_SVR)

svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
          np.array(training_labels))

support_vectors = np.transpose(svm.getSupportVectors())
rho, _, _ = svm.getDecisionFunction(0)
svm_detector = np.append(support_vectors, [[-rho]], 0)
hog.setSVMDetector(svm_detector)

def is_inside(i, o):
    ix, iy, iw, ih = i
    ox, oy, ow, oh = o
    return ix > ox and ix + iw < ox + ow and \
        iy > oy and iy + ih < oy + oh

for test_img_path in ['CarData/TestImages/test-0.pgm',
                      'CarData/TestImages/test-1.pgm',
                      '../images/car.jpg',
                      '../images/haying.jpg',
                      '../images/statue.jpg',
                      '../images/woodcutters.jpg']:
    img = cv2.imread(test_img_path)

    if OPENCV_MAJOR_VERSION >= 5 or \
            (OPENCV_MAJOR_VERSION == 4 and OPENCV_MINOR_VERSION >= 6):
        # OpenCV 4.6 or a later version is being used.
        found_rects, found_weights = hog.detectMultiScale(
            img, winStride=(8, 8), scale=1.03, groupThreshold=2.0)
    else:
        # OpenCV 4.5 or an earlier version is being used.
        # The groupThreshold parameter used to be named finalThreshold.
        found_rects, found_weights = hog.detectMultiScale(
            img, winStride=(8, 8), scale=1.03, finalThreshold=2.0)

    found_rects_filtered = []
    found_weights_filtered = []
    for ri, r in enumerate(found_rects):
        if found_weights[ri] < HOG_WEIGHT_THRESHOLD:
            continue
        for qi, q in enumerate(found_rects):
            if found_weights[qi] < HOG_WEIGHT_THRESHOLD:
                continue
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

    cv2.imshow(test_img_path, img)
cv2.waitKey(0)
