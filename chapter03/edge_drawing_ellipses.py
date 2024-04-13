import random as rng

import cv2
import numpy as np


planets = cv2.imread("../images/planet_glow.jpg")
gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
gray_img = cv2.medianBlur(gray_img, 5)

segments_viz = planets.copy()
ellipses_viz = planets.copy()

edge_drawing = cv2.ximgproc.createEdgeDrawing()

# Detect edges and get the resulting edge segments.
edge_drawing.detectEdges(gray_img)
segments = edge_drawing.getSegments()

# Detect circles and ellipses based on the detected edges.
ellipses = edge_drawing.detectEllipses()

# Draw the detected edge segments.
for segment in segments:
    color = (rng.randint(16, 256),
             rng.randint(16, 256),
             rng.randint(16, 256))
    cv2.polylines(segments_viz, [segment], False, color, 1,
                  cv2.LINE_8)

# Draw the detected circles and ellipses.
if ellipses is not None:
    for ellipse in ellipses:
        ellipse = ellipse.squeeze()
        center = (int(ellipse[0]), int(ellipse[1]))
        axes = (int(ellipse[2] + ellipse[3]),
                int(ellipse[2] + ellipse[4]))
        angle = ellipse[5]
        if ellipse[2] == 0:  # Ellipse
            color = (0, 0, 255)
        else:  # Circle
            color = (0, 255, 0)
        cv2.ellipse(ellipses_viz, center, axes, angle, 0, 360,
                    color, 2, cv2.LINE_AA)

cv2.imwrite("planets_edge_drawing_segments.jpg", segments_viz)
cv2.imwrite("planets_edge_drawing_ellipses.jpg", ellipses_viz)

cv2.imshow("Detected edge segments", segments_viz)
cv2.imshow("Detected circles (green) and ellipses (red)", ellipses_viz)

cv2.waitKey()
cv2.destroyAllWindows()
