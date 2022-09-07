import random as rng

import cv2
import numpy as np


img = cv2.imread("../images/statue_small.jpg",
                 cv2.IMREAD_GRAYSCALE)

h, w = img.shape
viz = np.empty((h, w, 3), dtype=np.uint8)

edge_drawing = cv2.ximgproc.createEdgeDrawing()

# Detect edges and get the resulting edge segments.
edge_drawing.detectEdges(img)
segments = edge_drawing.getSegments()

# Draw the detected edge segments.
for segment in segments:
    color = (rng.randint(16, 256),
             rng.randint(16, 256),
             rng.randint(16, 256))
    cv2.polylines(viz, [segment], False, color, 1, cv2.LINE_8)

cv2.imshow("Detected edge segments", viz)

cv2.waitKey()
cv2.destroyAllWindows()
