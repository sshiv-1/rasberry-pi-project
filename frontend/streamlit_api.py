from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests


HEALTH_PATH = "/healthz"
PREDICT_PATH = "/api/predict"
HEALTH_TIMEOUT = (3, 8)
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

    # Allow "192.168.1.20:5000" or "raspberrypi.local:5000" without a scheme.
    if "://" not in value:
        value = f"http://{value}"

    parsed = urlsplit(value)
    path = parsed.path.rstrip("/")
    for endpoint in (PREDICT_PATH, HEALTH_PATH):
        if path.endswith(endpoint):
            path = path[: -len(endpoint)].rstrip("/")
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def endpoint_url(base_url: str, path: str) -> str:
    base = normalize_base_url(base_url)
    if not base:
        raise RuntimeError("Enter the Raspberry Pi server address first.")
    return f"{base}{path}"


def _send(method: str, endpoint: str, **kwargs: Any) -> requests.Response:
    """Send a request and turn low-level network errors into readable ones."""
    try:
        request_method = getattr(requests, method.casefold())
        return request_method(endpoint, **kwargs)
    except requests.exceptions.ConnectTimeout as exc:
        raise RuntimeError(
            f"Timed out connecting to the Raspberry Pi. Check the address "
            f"and that it's powered on. Endpoint: {endpoint}"
        ) from exc
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            f"The Raspberry Pi connected but took too long to respond. "
            f"Endpoint: {endpoint}"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Request to {endpoint} failed: {exc}") from exc


def response_json(response: requests.Response, endpoint: str) -> dict[str, Any]:
    content_type = response.headers.get("Content-Type", "").casefold()
    if "json" not in content_type:
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


def _error_text(payload: dict[str, Any]) -> str | None:
    """Pull a readable error string out of the payload, if there is one."""
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    if isinstance(error, dict):
        nested = error.get("message")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def get_health(base_url: str) -> dict[str, Any]:
    endpoint = endpoint_url(base_url, HEALTH_PATH)
    response = _send("GET", endpoint, timeout=HEALTH_TIMEOUT)
    payload = response_json(response, endpoint)
    if not response.ok or payload.get("status") != "ok":
        raise RuntimeError(
            _error_text(payload)
            or f"Health check failed with HTTP {response.status_code}."
        )
    return payload


def api_status_message(status_code: int, endpoint: str) -> str:
    messages = {
        400: "The Raspberry Pi couldn't understand the request.",
        404: "The Raspberry Pi prediction endpoint was not found.",
        413: "The converted image is too large for the Raspberry Pi.",
        422: "The Raspberry Pi rejected the uploaded image.",
        500: "The Raspberry Pi failed while running inference.",
        502: "The Raspberry Pi gateway returned a bad response.",
        503: "The Raspberry Pi is busy or the model isn't ready yet.",
        504: "The Raspberry Pi gateway timed out.",
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
    if not jpeg_bytes:
        raise RuntimeError("No image data to send.")

    endpoint = endpoint_url(base_url, PREDICT_PATH)
    response = _send(
        "POST",
        endpoint,
        files={"leaf_image": (filename, jpeg_bytes, "image/jpeg")},
        timeout=REQUEST_TIMEOUT,
    )
    payload = response_json(response, endpoint)
    if not response.ok or payload.get("success") is not True:
        message = _error_text(payload) or api_status_message(
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