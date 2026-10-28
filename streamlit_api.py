from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests


HEALTH_PATH = "/healthz"
PREDICT_PATH = "/api/predict"
REQUEST_TIMEOUT = (5, 90)


class PredictionResponseError(RuntimeError):
    """A prediction response that includes the backend's structured error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        payload: dict[str, Any],
        endpoint: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload
        self.endpoint = endpoint


def normalize_base_url(base_url: str) -> str:
    """Normalize a user-entered server URL without duplicating API paths."""
    value = base_url.strip().rstrip("/")
    if not value:
        return value

    parsed = urlsplit(value)
    path = parsed.path.rstrip("/")
    for endpoint in (PREDICT_PATH, HEALTH_PATH):
        if path.endswith(endpoint):
            path = path[: -len(endpoint)].rstrip("/")
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def endpoint_url(base_url: str, path: str) -> str:
    return f"{normalize_base_url(base_url)}{path}"


def response_json(response: requests.Response, endpoint: str) -> dict[str, Any]:
    content_type = response.headers.get("Content-Type", "").casefold()
    if "application/json" not in content_type:
        raise RuntimeError(
            f"The Raspberry Pi returned a non-JSON response "
            f"(HTTP {response.status_code}) from {endpoint}."
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"The Raspberry Pi returned invalid JSON "
            f"(HTTP {response.status_code}) from {endpoint}."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"The Raspberry Pi returned an unexpected JSON payload "
            f"(HTTP {response.status_code}) from {endpoint}."
        )
    return payload


def get_health(base_url: str) -> dict[str, Any]:
    endpoint = endpoint_url(base_url, HEALTH_PATH)
    response = requests.get(endpoint, timeout=(3, 8))
    payload = response_json(response, endpoint)
    if not response.ok or payload.get("status") != "ok":
        raise RuntimeError(
            payload.get("error")
            or f"Health check failed with HTTP {response.status_code}."
        )
    return payload


def api_status_message(status_code: int, endpoint: str) -> str:
    messages = {
        404: "The Raspberry Pi prediction endpoint was not found.",
        413: "The converted image is too large for the Raspberry Pi.",
        422: "The Raspberry Pi rejected the uploaded image.",
        500: "The Raspberry Pi failed while running inference.",
    }
    return messages.get(
        status_code,
        f"Prediction request failed with HTTP {status_code}.",
    ) + f" Endpoint: {endpoint}"


def post_prediction(
    base_url: str,
    jpeg_bytes: bytes,
    filename: str = "leaf.jpg",
) -> dict[str, Any]:
    endpoint = endpoint_url(base_url, PREDICT_PATH)
    response = requests.post(
        endpoint,
        files={"leaf_image": (filename, jpeg_bytes, "image/jpeg")},
        timeout=REQUEST_TIMEOUT,
    )
    payload = response_json(response, endpoint)
    if not response.ok or payload.get("success") is not True:
        message = payload.get("error") or api_status_message(
            response.status_code,
            endpoint,
        )
        raise PredictionResponseError(
            f"{message} (HTTP {response.status_code}).",
            status_code=response.status_code,
            payload=payload,
            endpoint=endpoint,
        )
    return payload
