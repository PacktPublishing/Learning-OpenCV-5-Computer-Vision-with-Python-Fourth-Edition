import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import landmark_utils as u

# Download hand_landmarker.task from the MediaPipe model gallery:
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def main():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    number = 0

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            receivedKey = cv2.waitKey(20)
            number = (receivedKey - 48) if receivedKey != -1 else -1

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            results = landmarker.detect(mp_image)

            if results.hand_landmarks and number in [0, 1, 2, 3]:
                for hand_landmarks in results.hand_landmarks:
                    landmark_list = u.calc_landmark_list(image, hand_landmarks)
                    pre_processed_landmark_list = u.pre_process_landmark(
                        landmark_list)
                    u.log_csv(number, pre_processed_landmark_list)

                    h, w = image.shape[:2]
                    points = [(int(lm.x * w), int(lm.y * h))
                              for lm in hand_landmarks]
                    for x, y in points:
                        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                    for start, end in HAND_CONNECTIONS:
                        cv2.line(image, points[start], points[end],
                                 (0, 255, 0), 2)

            final = cv2.flip(image, 1)
            text = ""
            if number == -1:
                text = "Press key for gesture number"
            else:
                text = "Gesture: {}".format(number)
            cv2.putText(final, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', final)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    main()
