# hand gesture recognition with OpenCV and MediaPipe

This is a simple example of how landmarks identified by mediapipe can be used as training data for gesture recognition.

## Usage

Run `python train.py`, you will see webcam input being displayed.
If you press any key between 0 and 9 (and keep it pressed) you will generate training data for a hand post labeled with the number you are pressing.
Eg. do an "OK" hand in front of the webcam and press 0, move your hand around to generate diversified training data, release the number key)

Repeate the process for other gestures. In my example I generated data for Open hand (0), Thumb Up (1), OK (2) and Peace (3).

You then run the classify_gestures.ipynb notebook with jupyter notebooks.
It will create and train a small neural network. Default is 4 classes but you can change that (but make sure to provide training data as well in the step above).

Once you are done with generating training data and creating the classifier, you can test it with detect_gesture.py , the prediction will be dislayed in the top left corner.

Happy training.

# Recognition

Utility functions for landmark processing for csv storage and baseline for deep learning model taken from [Nikita Kiselov](https://github.com/kinivi) in [this repo](https://github.com/kinivi/hand-gesture-recognition-mediapipe) in turn translated from [this repo by Kazuhito Takahashi](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
