from __future__ import annotations

from io import BytesIO
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from streamlit_api import (
    PREDICT_PATH,
    PredictionResponseError,
    endpoint_url,
    get_health,
    normalize_base_url,
    post_prediction,
)


st.set_page_config(
    page_title="LeafLens · Plant Health Scanner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_PI_API_URL = os.environ.get("PI_API_URL", "http://127.0.0.1:5000")
MAX_FRONTEND_IMAGE_MB = 10
MAX_FRONTEND_IMAGE_BYTES = MAX_FRONTEND_IMAGE_MB * 1024 * 1024
MAX_FRONTEND_IMAGE_PIXELS = 20_000_000

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
        --leaflens-forest: #174b35;
        --leaflens-green: #27764c;
        --leaflens-lime: #d7f36a;
        --leaflens-line: color-mix(in srgb, var(--secondary-background-color) 75%, var(--text-color));
    }
    html, body, [class*="st-"], p, label, input, button, textarea {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 7% 0%, color-mix(in srgb, #b9dca0 38%, var(--background-color)) 0%, transparent 36%),
            var(--background-color);
        color: var(--text-color);
    }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text-color) !important;
        letter-spacing: -.035em;
    }
    p, label, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {
        color: var(--text-color);
    }
    .hero {
        background: var(--leaflens-forest);
        color: #f7fbf1;
        padding: 2.25rem 2.5rem;
        border-radius: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: '✳';
        position: absolute;
        right: 2.2rem;
        top: -2.4rem;
        color: var(--leaflens-lime);
        opacity: .9;
        font-size: 11rem;
        line-height: 1;
        transform: rotate(14deg);
    }
    .hero-kicker, .hero-title, .hero-copy { position: relative; z-index: 1; }
    .hero-kicker {
        color: var(--leaflens-lime);
        text-transform: uppercase;
        letter-spacing: .18em;
        font-size: .73rem;
        font-weight: 800;
    }
    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.2rem, 5vw, 4rem);
        font-weight: 800;
        letter-spacing: -.065em;
        line-height: 1.02;
        margin: .65rem 0 .75rem;
        max-width: 720px;
    }
    .hero-copy { color: #d4e4d6; max-width: 700px; font-size: 1.02rem; line-height: 1.65; }
    .step-label {
        color: var(--leaflens-green);
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .soft-card {
        background: var(--secondary-background-color);
        border: 1px solid var(--leaflens-line);
        border-radius: 22px;
        padding: 1.25rem 1.4rem;
    }
    .soft-card p { color: var(--text-color); }
    .status-label {
        display: inline-block;
        padding: .34rem .7rem;
        border-radius: 999px;
        background: color-mix(in srgb, var(--leaflens-green) 16%, var(--secondary-background-color));
        color: var(--text-color);
        font-size: .82rem;
        font-weight: 800;
    }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid var(--leaflens-line);
        padding: 1rem;
        border-radius: 16px;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
    }
    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 2.8rem;
    }
    div.stButton > button[kind="primary"] {
        background: var(--leaflens-green);
        border: 1px solid var(--leaflens-green);
        color: #fff;
    }
    div.stButton > button[kind="primary"]:hover {
        background: var(--leaflens-forest);
        border-color: var(--leaflens-forest);
    }
    div.stButton > button:disabled {
        opacity: .55;
        color: var(--text-color);
        border-color: var(--leaflens-line);
    }
    [data-testid="stFileUploader"] {
        background: var(--secondary-background-color);
        border: 1px solid var(--leaflens-line);
        border-radius: 16px;
        padding: .65rem;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] button {
        color: var(--text-color);
    }
    [data-testid="stAlert"] { color: var(--text-color); }
    [data-testid="stExpander"] {
        background: var(--secondary-background-color);
        border: 1px solid var(--leaflens-line);
        border-radius: 14px;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--leaflens-line); }
    .footer-note { color: var(--text-color); opacity: .72; font-size: .82rem; text-align: center; padding-top: 1.5rem; }

    /* --- refinements --- */
    .hero:after { pointer-events: none; }
    @media (max-width: 640px) {
        .hero { padding: 1.5rem 1.25rem; border-radius: 22px; }
        .hero:after { font-size: 7rem; right: -.5rem; top: -1.5rem; }
        .block-container { padding-top: 1rem; }
    }
    .soft-card p:last-child { margin-bottom: 0; }
    div.stButton > button { transition: background .15s ease, transform .1s ease; }
    div.stButton > button[kind="primary"]:active { transform: scale(.99); }
    [data-testid="stImage"] img { border-radius: 16px; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=8)
def to_jpeg_bytes(file_bytes: bytes) -> tuple[bytes, Image.Image]:
    """Validate, orient, resize, and encode one uploaded image in memory."""
    if not file_bytes:
        raise ValueError("The selected file is empty.")
    if len(file_bytes) > MAX_FRONTEND_IMAGE_BYTES:
        raise ValueError(
            f"The selected file is larger than {MAX_FRONTEND_IMAGE_MB} MB."
        )

    with Image.open(BytesIO(file_bytes)) as opened:
        opened.verify()

    with Image.open(BytesIO(file_bytes)) as opened:
        if opened.width * opened.height > MAX_FRONTEND_IMAGE_PIXELS:
            raise ValueError(
                "The selected image has too many pixels. Choose a smaller photo."
            )
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((2200, 2200), resample=Image.Resampling.LANCZOS)
        preview = image.copy()
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)

    jpeg_bytes = buffer.getvalue()
    if len(jpeg_bytes) > MAX_FRONTEND_IMAGE_BYTES:
        raise ValueError(
            f"The converted image exceeds {MAX_FRONTEND_IMAGE_MB} MB. "
            "Try a smaller photo."
        )
    return jpeg_bytes, preview


def set_disconnected(message: str) -> None:
    st.session_state["pi_connection_ok"] = False
    st.session_state.pop("leaflens_result", None)
    st.session_state["leaflens_error"] = message
    st.session_state.pop("leaflens_error_payload", None)


def analyze_file(uploaded_file) -> None:
    if uploaded_file is None:
        return

    with st.spinner("Checking the Raspberry Pi, then running inference…"):
        try:
            get_health(st.session_state["pi_api_url"])
            jpeg_bytes, _preview = to_jpeg_bytes(uploaded_file.getvalue())
            result = post_prediction(
                st.session_state["pi_api_url"],
                jpeg_bytes,
                "leaf.jpg",
            )
        except PredictionResponseError as exc:
            st.session_state["pi_connection_ok"] = True
            st.session_state.pop("leaflens_result", None)
            st.session_state["leaflens_error"] = (
                "The Raspberry Pi rejected this image: "
                f"{exc.payload.get('error', str(exc))}"
            )
            st.session_state["leaflens_error_payload"] = exc.payload
        except (requests.RequestException, RuntimeError) as exc:
            set_disconnected(
                "Raspberry Pi inference is unavailable. "
                f"Details: {exc}"
            )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.session_state.pop("leaflens_result", None)
            st.session_state["leaflens_error"] = f"Could not prepare that image: {exc}"
            st.session_state.pop("leaflens_error_payload", None)
        else:
            st.session_state["pi_connection_ok"] = True
            st.session_state["leaflens_result"] = result
            st.session_state.pop("leaflens_error", None)
            st.session_state.pop("leaflens_error_payload", None)


def render_precheck(payload: dict[str, Any]) -> None:
    leaf_check = payload.get("leaf_check")
    if not isinstance(leaf_check, dict) or not leaf_check:
        st.info(
            "The Raspberry Pi response did not include leaf-likeness details. "
            "This frontend does not run a local leaf detector."
        )
        return

    passed = leaf_check.get("passed")
    status = "PASS" if passed is True else "NOT PASSED" if passed is False else "NOT REPORTED"
    score = leaf_check.get("score")
    method = leaf_check.get("method") or "Backend method not reported"
    explanation = leaf_check.get("message") or "The backend did not provide an explanation."

    check_cols = st.columns([0.32, 0.68])
    with check_cols[0]:
        st.metric("Heuristic result", status)
        if score is not None:
            try:
                st.metric("Heuristic score", f"{float(score):.3f}")
            except (TypeError, ValueError):
                st.metric("Heuristic score", str(score))
    with check_cols[1]:
        st.info(explanation)
        st.caption(f"Method: {method} · This is a colour/contrast heuristic, not object detection.")
        if passed is True:
            st.caption("Passing this heuristic does not guarantee that an actual leaf is present.")


def render_results(result: dict[str, Any]) -> None:
    st.divider()
    st.markdown('<div class="step-label">Step 04 · Results</div>', unsafe_allow_html=True)
    st.title("Analysis results")

    st.markdown("### Leaf-likeness pre-check")
    render_precheck(result)

    prediction = result.get("prediction") or {}
    predicted_class = prediction.get("class_name", "Unknown class")
    display_name = prediction.get("display_label", predicted_class)
    confidence = prediction.get("confidence", 0.0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    is_healthy = bool(prediction.get("is_healthy", "healthy" in predicted_class.casefold()))
    top_predictions = result.get("top_predictions") or []
    advice = result.get("treatment")
    advice = advice if isinstance(advice, dict) else {}

    outcome_cols = st.columns([1.35, 0.65], gap="large")
    with outcome_cols[0]:
        st.markdown("### Most likely class")
        st.subheader(display_name)
        st.caption(f"Exact model class key: `{predicted_class}`")
        badge = "✅ Healthy class predicted" if is_healthy else "⚠️ Disease/condition class predicted"
        st.markdown(f'<span class="status-label">{badge}</span>', unsafe_allow_html=True)
        st.write("")
        if result.get("low_confidence"):
            st.warning("Low confidence. Predictions are preliminary and confidence is not proof of a confirmed diagnosis.")
        else:
            st.success("Prediction returned by the Raspberry Pi.")
    with outcome_cols[1]:
        st.metric("Model confidence", f"{confidence_value * 100:.1f}%")
        st.progress(min(max(confidence_value, 0.0), 1.0), text="Top-class score")

    st.markdown("### Top three predictions")
    rows = []
    for rank, item in enumerate(top_predictions[:3], start=1):
        if not isinstance(item, dict):
            continue
        label = item.get("display_label", item.get("label", item.get("class_name", "Unknown")))
        try:
            item_confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            item_confidence = 0.0
        rows.append({"#": rank, "Class": label, "Confidence": item_confidence * 100})
    if rows:
        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%.2f%%"
                ),
            },
        )
    else:
        st.warning("The Raspberry Pi did not return top predictions.")

    st.markdown("### Treatment advice")
    advice_items = advice.get("advice")
    if isinstance(advice_items, list) and all(isinstance(item, str) and item.strip() for item in advice_items):
        for item in advice_items:
            st.markdown(f"- {item}")
        if advice.get("note"):
            st.caption(str(advice["note"]))
    else:
        st.warning(
            f"No usable advice was returned for predicted class `{predicted_class}`. "
            "The frontend does not invent disease-specific guidance."
        )

    with st.expander("Technical details"):
        leaf_check = result.get("leaf_check")
        leaf_check = leaf_check if isinstance(leaf_check, dict) else {}
        st.json({
            "raspberry_pi_api": endpoint_url(st.session_state["pi_api_url"], PREDICT_PATH),
            "predicted_class_key": predicted_class,
            "advice_returned": bool(
                isinstance(advice_items, list)
                and any(isinstance(item, str) and item.strip() for item in advice_items)
            ),
            "advice_item_count": (
                len(advice_items) if isinstance(advice_items, list) else 0
            ),
            "leaf_check_method": leaf_check.get("method"),
            "leaf_likeness_score": leaf_check.get("score"),
        })


def render_failed_precheck(payload: dict[str, Any]) -> None:
    if isinstance(payload.get("leaf_check"), dict):
        st.markdown("### Leaf-likeness pre-check")
        render_precheck(payload)


with st.sidebar:
    st.markdown("## 🌿 LeafLens")
    st.caption("Upload-only frontend · Raspberry Pi inference")
    pi_api_url = normalize_base_url(
        st.text_input(
            "Raspberry Pi server URL",
            value=st.session_state.get("pi_api_url", DEFAULT_PI_API_URL),
            placeholder="http://192.168.1.50:5000",
            help="Use the Pi's LAN IP and port 5000. Both devices should be on the same Wi-Fi.",
        )
    )
    if pi_api_url != st.session_state.get("pi_api_url"):
        st.session_state["pi_connection_ok"] = False
        st.session_state.pop("leaflens_result", None)
    st.session_state["pi_api_url"] = pi_api_url

    if st.button("Test Raspberry Pi connection", use_container_width=True):
        try:
            health = get_health(pi_api_url)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            set_disconnected(
                "Raspberry Pi is unavailable. Inference requires a reachable Pi. "
                f"Details: {exc}"
            )
        else:
            st.session_state["pi_connection_ok"] = True
            st.session_state.pop("leaflens_error", None)
            st.success("Raspberry Pi is reachable")
            st.write(f"Classes: **{health.get('classes', 'unknown')}**")
            st.write(f"Input size: **{health.get('image_size', 'unknown')} px**")

    if st.session_state.get("pi_connection_ok"):
        st.success("Ready for Raspberry Pi analysis")
    else:
        st.warning("Disconnected — test the Raspberry Pi connection before analyzing.")

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("1. Upload a photo\n2. Convert it to JPEG in memory\n3. Check `/healthz`\n4. Send it to the Raspberry Pi\n5. Show returned classification and advice")


st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Plant health · Raspberry Pi AI</div>
      <div class="hero-title">A clearer picture of plant health.</div>
      <div class="hero-copy">Upload a photo. The image is converted to JPEG in memory and sent to the Raspberry Pi, where the ONNX model returns the classification and care guidance.</div>
    </section>
    """,
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1.1, 0.9], gap="large")
with left:
    st.markdown('<div class="step-label">Step 01 · Upload image</div>', unsafe_allow_html=True)
    st.subheader("Choose a leaf photo")
    uploaded_photo = st.file_uploader(
        "Upload a JPG, JPEG, PNG or WebP image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key="leaflens_upload_input",
        help=f"Maximum file size: {MAX_FRONTEND_IMAGE_MB} MB.",
    )
    # Clear stale results when a different image is uploaded
    upload_id = (
        getattr(uploaded_photo, "file_id", None) or (uploaded_photo.name, uploaded_photo.size)
        if uploaded_photo is not None else None
    )
    if upload_id != st.session_state.get("leaflens_upload_id"):
        st.session_state["leaflens_upload_id"] = upload_id
        for key in ("leaflens_result", "leaflens_error", "leaflens_error_payload"):
            st.session_state.pop(key, None)

    preview_ready = uploaded_photo is not None
    if uploaded_photo is not None:
        try:
            _, upload_preview = to_jpeg_bytes(uploaded_photo.getvalue())
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            preview_ready = False
            st.error(f"This file could not be opened as an image: {exc}")
        else:
            st.image(upload_preview, caption="Selected image", use_container_width=True)
    analyze_disabled = not (
        preview_ready and st.session_state.get("pi_connection_ok", False)
    )
    if st.button(
        "Analyze Leaf",
        type="primary",
        use_container_width=True,
        disabled=analyze_disabled,
        key="analyze_upload_button",
    ):
        analyze_file(uploaded_photo)

with right:
    st.markdown('<div class="step-label">Step 02 · Raspberry Pi</div>', unsafe_allow_html=True)
    st.subheader("Inference stays on the Pi")
    st.markdown(
        """
        <div class="soft-card">
          <p><strong>Upload</strong><br/>Choose one readable leaf image from your device.</p>
          <p><strong>Connection</strong><br/>The frontend checks the Raspberry Pi before every analysis.</p>
          <p><strong>Inference</strong><br/>The Pi runs the ONNX model and returns the class, confidence, pre-check details when available, and backend advice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Inference is unavailable while the Raspberry Pi is disconnected.")
    st.caption("The frontend does not run ONNX, PyTorch or TensorFlow and does not save uploads intentionally.")


if st.session_state.get("leaflens_error"):
    error_payload = st.session_state.get("leaflens_error_payload")
    if isinstance(error_payload, dict):
        st.warning(st.session_state["leaflens_error"])
        render_failed_precheck(error_payload)
    else:
        st.error(st.session_state["leaflens_error"])

result = st.session_state.get("leaflens_result")
if isinstance(result, dict):
    render_results(result)

st.markdown(
    '<div class="footer-note">LeafLens is a screening aid, not a confirmed diagnosis. Confirm significant symptoms with a local agricultural expert.</div>',
    unsafe_allow_html=True,
)