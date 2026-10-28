import json
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
sys.path.insert(0, str(FRONTEND_DIR))


def json_response(status: int, payload: dict, content_type: str = "application/json"):
    response = requests.Response()
    response.status_code = status
    response.headers["Content-Type"] = content_type
    response._content = json.dumps(payload).encode()
    return response


def test_frontend_normalizes_endpoint_suffixes():
    from streamlit_api import endpoint_url, normalize_base_url

    assert normalize_base_url("http://127.0.0.1:5000/") == "http://127.0.0.1:5000"
    assert normalize_base_url("http://127.0.0.1:5000/healthz") == "http://127.0.0.1:5000"
    assert normalize_base_url("http://127.0.0.1:5000/api/predict") == "http://127.0.0.1:5000"
    assert endpoint_url("http://127.0.0.1:5000/", "/api/predict") == (
        "http://127.0.0.1:5000/api/predict"
    )


def test_frontend_posts_canonical_multipart_field(monkeypatch):
    from streamlit_api import post_prediction

    captured = {}

    def fake_post(url, *, files, timeout):
        captured.update(url=url, files=files, timeout=timeout)
        return json_response(200, {"success": True, "prediction": {}})

    monkeypatch.setattr("streamlit_api.requests.post", fake_post)

    result = post_prediction("http://127.0.0.1:5000/", b"jpeg", "leaf.jpg")

    assert result["success"] is True
    assert captured["url"] == "http://127.0.0.1:5000/api/predict"
    assert captured["files"]["leaf_image"] == ("leaf.jpg", b"jpeg", "image/jpeg")


def test_frontend_handles_html_error_without_json_traceback(monkeypatch):
    from streamlit_api import post_prediction

    def fake_post(url, *, files, timeout):
        return json_response(404, {}, "text/html")

    monkeypatch.setattr("streamlit_api.requests.post", fake_post)

    with pytest.raises(RuntimeError, match="non-JSON response \\(HTTP 404\\)"):
        post_prediction("http://127.0.0.1:5000", b"jpeg")


def test_frontend_preserves_structured_prediction_error(monkeypatch):
    from streamlit_api import PredictionResponseError, post_prediction

    def fake_post(url, *, files, timeout):
        return json_response(
            422,
            {
                "success": False,
                "error": "The heuristic rejected the image.",
                "leaf_check": {
                    "passed": False,
                    "score": 0.12,
                    "message": "Insufficient green contrast.",
                    "method": "colour_contrast",
                },
            },
        )

    monkeypatch.setattr("streamlit_api.requests.post", fake_post)

    with pytest.raises(PredictionResponseError) as raised:
        post_prediction("http://127.0.0.1:5000", b"jpeg")

    assert raised.value.status_code == 422
    assert raised.value.payload["leaf_check"]["passed"] is False


def test_frontend_preserves_disconnected_backend_error(monkeypatch):
    from streamlit_api import post_prediction

    def fake_post(url, *, files, timeout):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("streamlit_api.requests.post", fake_post)

    with pytest.raises(requests.ConnectionError, match="connection refused"):
        post_prediction("http://127.0.0.1:5000", b"jpeg")


def test_streamlit_uses_canonical_contract():
    source = (FRONTEND_DIR / "streamlit_app.py").read_text(encoding="utf-8")

    assert "http://127.0.0.1:5000" in source
    assert "PREDICT_PATH" in source
    assert "post_prediction" in source


def test_streamlit_frontend_is_upload_only_and_has_no_local_advice():
    source = (FRONTEND_DIR / "streamlit_app.py").read_text(encoding="utf-8")

    assert "st.camera_input" not in source
    assert "st.tabs" not in source
    assert "treatment_advice" not in source
    assert 'type=["jpg", "jpeg", "png", "webp"]' in source
