import cv2
import numpy as np


img = cv2.imread('../images/houghlines5.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edge_drawing = cv2.ximgproc.createEdgeDrawing()

edge_drawing_params = cv2.ximgproc_EdgeDrawing_Params()
edge_drawing_params.MinLineLength = 20

edge_drawing.setParams(edge_drawing_params)

# Detect edges.
edge_drawing.detectEdges(gray_img)

# Detect lines based on the edges and the specified parameters.
lines = edge_drawing.detectLines()

# Draw the detected lines.
if lines is not None:
    lines = np.uint16(np.around(lines))
    for line in lines:
        line = line.squeeze()
        cv2.line(img, (line[0], line[1]),
        (line[2], line[3]), (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow("Detected lines", img)
cv2.waitKey()
cv2.destroyAllWindows()
