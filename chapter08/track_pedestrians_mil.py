import cv2
import numpy as np

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

class Pedestrian():
    """A tracked pedestrian with a state including an ID, tracking
    window, histogram, and Kalman filter.
    """

    def __init__(self, id, frame, track_window):

        self.id = id

        # Initialize the MIL tracker.
        self.tracker = cv2.TrackerMIL_create()
        self.tracker.init(frame, track_window)

    def update(self, frame):

        # Update the MIL tracker.
        ret, (x, y, w, h) = self.tracker.update(frame)

        # Draw the corrected tracking window as a cyan rectangle.
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

        # Draw the ID above the rectangle in blue text.
        cv2.putText(frame, 'ID: %d' % self.id, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                    1, cv2.LINE_AA)

def main():

    cap = cv2.VideoCapture('pedestrians.avi')

    # Create the KNN background subtractor.
    bg_subtractor = cv2.createBackgroundSubtractorKNN()
    history_length = 20
    bg_subtractor.setHistory(history_length)

    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (8, 3))

    pedestrians = []
    num_history_frames_populated = 0
    while True:
        grabbed, frame = cap.read()
        if (grabbed is False):
            break

        # Apply the KNN background subtractor.
        fg_mask = bg_subtractor.apply(frame)

        # Let the background subtractor build up a history.
        if num_history_frames_populated < history_length:
            num_history_frames_populated += 1
            continue

        # Create the thresholded image.
        _, thresh = cv2.threshold(fg_mask, 127, 255,
                                  cv2.THRESH_BINARY)
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

        contours, hier = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw green rectangles around large contours.
        # Also, if no pedestrians are being tracked yet, create some.
        should_initialize_pedestrians = len(pedestrians) == 0
        id = 0
        for c in contours:
            if cv2.contourArea(c) > 500:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (0, 255, 0), 1)
                if should_initialize_pedestrians:
                    pedestrians.append(
                        Pedestrian(id, frame, (x, y, w, h)))
            id += 1

        # Update the tracking of each pedestrian.
        for pedestrian in pedestrians:
            pedestrian.update(frame)

        cv2.imshow('Pedestrians Tracked', frame)

        k = cv2.waitKey(110)
        if k == 27:  # Escape
            break

if __name__ == "__main__":
    main()
