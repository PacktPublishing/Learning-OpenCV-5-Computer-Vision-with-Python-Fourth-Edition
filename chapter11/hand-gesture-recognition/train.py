import copy
import csv
import itertools
import cv2
import numpy as np
import mediapipe as mp
import landmark_utils as u
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def main():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    number = 0
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            receivedKey = cv2.waitKey(20)
            number = receivedKey - 48
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks and number in [0, 1, 2, 3]:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = u.calc_landmark_list(image, hand_landmarks)
                    pre_processed_landmark_list = u.pre_process_landmark(
                        landmark_list)
                    u.log_csv(number, pre_processed_landmark_list)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

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


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


if __name__ == '__main__':
    main()
