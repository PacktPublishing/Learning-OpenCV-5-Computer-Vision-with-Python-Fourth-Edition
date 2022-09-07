import cv2
import numpy as np

img = cv2.imread("../images/statue_small.jpg",
                 cv2.IMREAD_GRAYSCALE)
cv2.imwrite("canny.jpg", cv2.Canny(img, 200, 300))
cv2.imshow("canny", cv2.imread("canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()
