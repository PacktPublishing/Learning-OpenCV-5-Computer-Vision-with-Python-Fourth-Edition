import json
import cv2
import requests

"""
Original version of lambda which just tests opencv
"""
"""
def lambda_handler(event, context):
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "hello OpenCV {} in a Lambda".format(cv2.__version__),
            }
        ),
    }
"""

"""
Detect faces on image from url
"""


def lambda_handler(event, context):
    image_url = event['queryStringParameters']['url']
    tmp_image = "/tmp/image.jpg"
    img_data = requests.get(image_url).content
    with open(tmp_image, 'wb') as handler:
        handler.write(img_data)
    img = cv2.imread(tmp_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("./cascade.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    coords = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
              for (x, y, w, h) in faces]
    return {
        "statusCode": 200,
        "body": json.dumps({"coords": coords}),
    }
