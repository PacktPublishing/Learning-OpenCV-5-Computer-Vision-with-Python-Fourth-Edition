import json
from unittest.mock import MagicMock, mock_open

import numpy as np
import pytest

from hello_world import app


@pytest.fixture()
def apigw_event():
    return {
        "queryStringParameters": {"url": "https://example.com/test.jpg"},
    }


def test_lambda_handler(apigw_event, monkeypatch):
    mock_response = MagicMock()
    mock_response.headers = {
        "Content-Type": "image/jpeg",
        "Content-Length": "5",
    }
    mock_response.iter_content.return_value = [b"image"]
    mock_response.raise_for_status.return_value = None

    mock_get = MagicMock(return_value=mock_response)
    monkeypatch.setattr(app.requests, "get", mock_get)
    monkeypatch.setattr("builtins.open", mock_open())
    monkeypatch.setattr(
        app.cv2,
        "imread",
        MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8)),
    )
    mock_cascade = MagicMock()
    mock_cascade.detectMultiScale.return_value = np.array([[10, 20, 30, 40]])
    monkeypatch.setattr(app, "cascade", mock_cascade)

    ret = app.lambda_handler(apigw_event, None)
    data = json.loads(ret["body"])

    assert ret["statusCode"] == 200
    assert data == {"coords": [{"x": 10, "y": 20, "w": 30, "h": 40}]}
    mock_get.assert_called_once_with(
        "https://example.com/test.jpg",
        timeout=app.DOWNLOAD_TIMEOUT,
        stream=True,
    )


@pytest.mark.parametrize(
    ("event", "expected_message"),
    [
        ({}, "missing required query string parameter 'url'"),
        ({"queryStringParameters": None}, "missing required query string parameter 'url'"),
        (
            {"queryStringParameters": {"url": "file:///tmp/image.jpg"}},
            "url must be an http(s) URL",
        ),
    ],
)
def test_lambda_handler_rejects_invalid_input(event, expected_message):
    ret = app.lambda_handler(event, None)

    assert ret["statusCode"] == 400
    assert json.loads(ret["body"]) == {"error": expected_message}


def test_lambda_handler_rejects_non_image(apigw_event, monkeypatch):
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr(app.requests, "get", MagicMock(return_value=mock_response))

    ret = app.lambda_handler(apigw_event, None)

    assert ret["statusCode"] == 400
    assert json.loads(ret["body"]) == {"error": "url did not return an image"}
