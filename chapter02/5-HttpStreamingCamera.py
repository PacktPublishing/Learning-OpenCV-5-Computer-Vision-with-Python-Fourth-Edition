import cv2
import flask


app = flask.Flask(__name__)


def mjpeg_generator():

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    success, frame = video_capture.read()
    while success:
        success, jpeg = cv2.imencode('.jpg', frame)
        if not success:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: %d\r\n'
               b'\r\n'
               b'%s\r\n' % (len(jpeg), jpeg.tobytes()))
        success, frame = video_capture.read()


@app.route('/')
def stream_from_camera():
    return flask.Response(
        mjpeg_generator(),
        mimetype='multipart/x-mixed-replace;boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
