import json
import os
from urllib.parse import urlparse

import cv2
import requests

ALLOWED_SCHEMES = {"http", "https"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DOWNLOAD_TIMEOUT = 5

CASCADE_PATH = os.path.join(os.path.dirname(__file__), "cascade.xml")
cascade = cv2.CascadeClassifier(CASCADE_PATH)
if cascade.empty():
    raise RuntimeError(f"Failed to load cascade classifier from {CASCADE_PATH}")


def _error(status_code, message):
    return {
        "statusCode": status_code,
        "body": json.dumps({"error": message}),
    }


def lambda_handler(event, context):
    image_url = (event.get("queryStringParameters") or {}).get("url")
    if not image_url:
        return _error(400, "missing required query string parameter 'url'")

    parsed = urlparse(image_url)
    if parsed.scheme not in ALLOWED_SCHEMES or not parsed.netloc:
        return _error(400, "url must be an http(s) URL")

    try:
        response = requests.get(image_url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        return _error(502, f"could not fetch image: {exc}")

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        return _error(400, "url did not return an image")

    content_length = int(response.headers.get("Content-Length", 0) or 0)
    if content_length > MAX_IMAGE_BYTES:
        return _error(413, "image exceeds maximum allowed size")

    tmp_image = "/tmp/image.jpg"
    bytes_written = 0
    with open(tmp_image, "wb") as handler:
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            bytes_written += len(chunk)
            if bytes_written > MAX_IMAGE_BYTES:
                return _error(413, "image exceeds maximum allowed size")
            handler.write(chunk)

    img = cv2.imread(tmp_image)
    if img is None:
        return _error(400, "downloaded file is not a readable image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    coords = [
        {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        for (x, y, w, h) in faces
    ]
    return {
        "statusCode": 200,
        "body": json.dumps({"coords": coords}),
    }
