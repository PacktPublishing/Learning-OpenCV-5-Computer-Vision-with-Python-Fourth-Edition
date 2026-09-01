import cv2
import numpy as np


model = cv2.dnn.readNetFromONNX('faces_data/yolo11n-face-age.onnx')
blob_length = 640
color_scale = 1.0/255.0
confidence_threshold = 0.5
labels = ['age: 0-14', 'age: 15-22', 'age: 22+']


cap = cv2.VideoCapture(0)

success, frame = cap.read()
while success:

    h, w = frame.shape[:2]

    # Pad the frame to make it square.
    padded_length = max(h, w)
    padded = np.zeros((padded_length, padded_length, 3), np.uint8)
    padded[0:h, 0:w] = frame

    # Calculate the scale factor of the frame relative to the blob.
    position_scale = padded_length / blob_length

    # Detect objects in the frame.

    blob = cv2.dnn.blobFromImage(
        padded, scalefactor=color_scale,
        size=(blob_length, blob_length), swapRB=True, crop=False)

    model.setInput(blob)
    results = model.forward()

    # Iterate over the detected objects.
    for cx, cy, w, h, *confidences in results[0].transpose():

        confidence = max(confidences)
        if confidence > confidence_threshold:

            # Scale the coordinates back to the frame size.
            x0 = int((cx-w/2) * position_scale)
            y0 = int((cy-h/2) * position_scale)
            x1 = int((cx+w/2) * position_scale)
            y1 = int((cy+h/2) * position_scale)

            # Get the class label.
            id = confidences.index(confidence)
            label = labels[id]

            # Draw a blue rectangle around the face.
            cv2.rectangle(frame, (x0, y0), (x1, y1),
                          (255, 0, 0), 2)

            # Draw the classification result and confidence.
            text = '%s (%.1f%%)' % (label, confidence * 100.0)
            cv2.putText(frame, text, (x0, y0 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Faces and age', frame)

    k = cv2.waitKey(1)
    if k == 27:  # Escape
        break

    success, frame = cap.read()
