import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


cap = cv2.VideoCapture(0)

base_options = python.BaseOptions(
    model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)

with vision.GestureRecognizer.create_from_options(options) as recognizer:

    while cap.isOpened():

        success, image = cap.read()
        if not success:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_image)

        results = recognizer.recognize(mp_image)
        gestures = results.gestures
        if len(gestures) > 0:
            best_gesture = gestures[0][0]
        else:
            best_gesture = None

        # Flip the image horizontally for a selfie view.
        final = cv2.flip(image, 1)

        if best_gesture:
            # Show the name of the gesture.
            cv2.putText(final, best_gesture.category_name,
                        (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

        cv2.imshow('MediaPipe Hands', final)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
