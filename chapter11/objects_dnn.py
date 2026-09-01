import cv2
import numpy as np


model = cv2.dnn.readNetFromONNX('objects_data/yolo11n.onnx')
blob_length = 640
color_scale = 1.0/255.0
confidence_threshold = 0.5
labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
          'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench',
          'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
          'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
          'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
          'snowboard', 'sports ball', 'kite', 'baseball bat',
          'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
          'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
          'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
          'donut', 'cake', 'chair', 'sofa', 'potted plant', 'bed',
          'dining table', 'toilet', 'TV or monitor', 'laptop',
          'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
          'oven', 'toaster', 'sink', 'refrigerator', 'book',
          'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
          'toothbrush']


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
    for x0, y0, x1, y1, confidence, id in results[0]:

        if confidence > confidence_threshold:

            # Scale the coordinates back to the frame size.
            x0 = int(x0 * position_scale)
            y0 = int(y0 * position_scale)
            x1 = int(x1 * position_scale)
            y1 = int(y1 * position_scale)

            # Get the class label.
            id = int(id)
            label = labels[id]

            # Draw a blue rectangle around the object.
            cv2.rectangle(frame, (x0, y0), (x1, y1),
                          (255, 0, 0), 2)

            # Draw the classification result and confidence.
            text = '%s (%.1f%%)' % (label, confidence * 100.0)
            cv2.putText(frame, text, (x0, y0 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Objects', frame)

    k = cv2.waitKey(1)
    if k == 27:  # Escape
        break

    success, frame = cap.read()
