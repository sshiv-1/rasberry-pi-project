import io
import json
import hashlib
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_json(name: str):
    with (ROOT / name).open(encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# -------------------------------------------------------------
# 1. ARTIFACT CONSISTENCY & STRUCTURE TESTS
# -------------------------------------------------------------

def test_class_names_and_treatment_keys_match():
    """Verify class_names.json and treatment_advice.json have exactly matching 38 keys."""
    classes = load_json("class_names.json")
    advice = load_json("treatment_advice.json")
    assert len(classes) == 38
    assert len(set(classes)) == 38, "Duplicate class names found"
    assert set(classes) == set(advice.keys()), "Keys mismatch between class names and treatment advice"


def test_corn_common_rust_uses_exact_model_label():
    """Regression test: Corn Common Rust has trailing underscore from PlantVillage naming."""
    classes = load_json("class_names.json")
    assert "Corn_(maize)___Common_rust_" in classes
    assert "Corn_(maize)___Common_rust" not in classes


def test_healthy_and_disease_advice_structure():
    """Validate structure and semantic correctness of treatment advice for both categories."""
    advice = load_json("treatment_advice.json")
    classes = load_json("class_names.json")

    healthy_count = 0
    disease_count = 0
    for name in classes:
        entry = advice[name]
        assert "label" in entry and isinstance(entry["label"], str) and len(entry["label"]) > 0
        assert "category" in entry and entry["category"] in ("healthy", "disease")
        assert "advice" in entry and isinstance(entry["advice"], list) and len(entry["advice"]) >= 1
        assert "note" in entry and isinstance(entry["note"], str)

        if "healthy" in name.lower():
            healthy_count += 1
            assert entry["category"] == "healthy"
            assert "no disease" in entry["advice"][0].lower()
        else:
            disease_count += 1
            assert entry["category"] == "disease"

    assert healthy_count == 12, f"Expected 12 healthy classes, found {healthy_count}"
    assert disease_count == 26, f"Expected 26 disease classes, found {disease_count}"


def test_deployment_metadata_consistency():
    """Verify deployment_metadata.json matches model.onnx and class_names.json hashes and shapes."""
    meta_path = ROOT / "deployment_metadata.json"
    assert meta_path.is_file(), "deployment_metadata.json must exist"
    meta = load_json("deployment_metadata.json")

    assert meta.get("architecture") == "MobileNetV2"
    assert meta.get("classes") == 38
    assert meta.get("input_size") == 160
    assert meta.get("backbone_frozen") is True
    assert meta.get("epochs_trained") == 50

    # Model and class hashes
    assert meta.get("model_sha256") == sha256_file(ROOT / "model.onnx")
    assert meta.get("class_names_sha256") == sha256_file(ROOT / "class_names.json")

    # Export metadata
    export = meta.get("export", {})
    assert export.get("input_layout") == "NCHW"
    assert export.get("input_dtype") == "float32"
    assert export.get("input_range") == "[0, 255]"
    assert export.get("dynamic_batch") is True


def test_model_artifact_is_present_and_standalone():
    """Verify model.onnx is present, standalone (>1MB), and requires no external data files."""
    model_path = ROOT / "model.onnx"
    assert model_path.is_file()
    assert model_path.stat().st_size > 1_000_000, "model.onnx should be >1MB"
    # Ensure no external data chunks are missing
    external_data = list(ROOT.glob("*.onnx.data"))
    assert len(external_data) == 0, "Model should be self-contained without external .onnx.data files"


# -------------------------------------------------------------
# 2. ONNX RUNTIME EXECUTION & PARITY CHECKS
# -------------------------------------------------------------

def test_onnx_shape_and_cpu_execution():
    """Verify ONNX input/output shapes, tensor types, and CPU forward pass."""
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    session = ort.InferenceSession(
        str(ROOT / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    assert input_meta.shape[1:] == [3, 160, 160], f"Unexpected input shape: {input_meta.shape}"
    assert input_meta.type == "tensor(float)"
    assert output_meta.shape[-1] == 38
    assert output_meta.type == "tensor(float)"

    # Execute with zero tensor
    output_zeros = session.run(
        [output_meta.name],
        {input_meta.name: np.zeros((1, 3, 160, 160), dtype=np.float32)},
    )[0]
    assert output_zeros.shape == (1, 38)
    assert np.all(output_zeros >= 0.0) and np.all(output_zeros <= 1.0)
    assert np.isclose(output_zeros.sum(), 1.0, atol=1e-3)

    # Execute with typical image values [0, 255]
    dummy_pixels = np.random.uniform(0.0, 255.0, size=(1, 3, 160, 160)).astype(np.float32)
    output_rand = session.run([output_meta.name], {input_meta.name: dummy_pixels})[0]
    assert output_rand.shape == (1, 38)
    assert np.all(output_rand >= 0.0) and np.all(output_rand <= 1.0)
    assert np.isclose(output_rand.sum(), 1.0, atol=1e-3)


# -------------------------------------------------------------
# 3. PREPROCESSING & EXIF HANDLING TESTS
# -------------------------------------------------------------

def test_image_preprocessing_pipeline():
    """Verify preprocess() handles RGB, RGBA, and Grayscale, producing contiguous (1, 3, 160, 160) float32 in [0, 255]."""
    pytest.importorskip("flask")
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    import sys
    sys.path.insert(0, str(ROOT))
    import pi_app
    import numpy as np

    # Test RGB image of unusual dimensions
    rgb_img = Image.new("RGB", (320, 240), color=(120, 150, 80))
    tensor = pi_app.preprocess(rgb_img)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 3, 160, 160)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert tensor.min() >= 0.0 and tensor.max() <= 255.0

    # Test RGBA image (transparency channel stripped to 3-channel RGB)
    rgba_img = Image.new("RGBA", (100, 100), color=(50, 100, 150, 200))
    tensor_rgba = pi_app.preprocess(rgba_img)
    assert tensor_rgba.shape == (1, 3, 160, 160)

    # Test Grayscale image (expanded to 3 channels)
    gray_img = Image.new("L", (80, 80), color=128)
    tensor_gray = pi_app.preprocess(gray_img)
    assert tensor_gray.shape == (1, 3, 160, 160)


def test_exif_orientation_handling():
    """Verify preprocess() correctly handles EXIF orientation metadata without crashing."""
    pytest.importorskip("PIL")
    import sys
    sys.path.insert(0, str(ROOT))
    import pi_app

    # Create image with EXIF tag for orientation 6 (90 degrees CW)
    img = Image.new("RGB", (200, 100), color="green")
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation tag: 6 = rotate 90 CW

    # Save to buffer and reopen to simulate phone upload with EXIF
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    uploaded = Image.open(buf)

    tensor = pi_app.preprocess(uploaded)
    assert tensor.shape == (1, 3, 160, 160)


# -------------------------------------------------------------
# 4. API & FLASK ENDPOINT BEHAVIORAL TESTS
# -------------------------------------------------------------

def make_test_image_bytes(format="JPEG", size=(160, 160), color="green") -> io.BytesIO:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf


def test_api_predict_json_success():
    """Verify POST /api/predict returns the canonical prediction schema."""
    pytest.importorskip("flask")
    import sys
    sys.path.insert(0, str(ROOT))
    import pi_app

    client = pi_app.app.test_client()
    img_buf = make_test_image_bytes()

    data = {"leaf_image": (img_buf, "test_leaf.jpg")}
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert response.is_json
    payload = response.get_json()

    assert payload["success"] is True
    assert "top_predictions" in payload
    assert isinstance(payload["top_predictions"], list)
    assert len(payload["top_predictions"]) == min(3, len(pi_app.class_names))

    top = payload["prediction"]
    assert "class_name" in top
    assert "display_label" in top
    assert "confidence" in top
    assert isinstance(top["is_healthy"], bool)
    assert 0.0 <= top["confidence"] <= 1.0

    assert "low_confidence" in payload
    assert isinstance(payload["low_confidence"], bool)
    assert payload["confidence_threshold"] == pi_app.MIN_CONFIDENCE

    advice = payload["treatment"]
    assert "advice" in advice
    assert "label" in advice
    assert "category" in advice


def test_healthz_endpoint():
    """Verify GET /healthz returns 200 and proper health status dictionary."""
    pytest.importorskip("flask")
    import sys
    sys.path.insert(0, str(ROOT))
    import pi_app

    client = pi_app.app.test_client()
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.is_json
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["model"] == "model.onnx"
    assert payload["classes"] == 38
    assert payload["image_size"] == 160
    assert payload["provider"] == "CPUExecutionProvider"


def test_api_invalid_and_corrupt_upload_returns_400():
    """Verify /api/predict returns JSON errors for missing and corrupt uploads."""
    pytest.importorskip("flask")
    import sys
    sys.path.insert(0, str(ROOT))
    import pi_app

    client = pi_app.app.test_client()

    # Empty payload
    r1 = client.post("/api/predict", data={})
    assert r1.status_code == 422
    assert r1.is_json
    assert "error" in r1.get_json()

    # Corrupt non-image bytes
    corrupt_buf = io.BytesIO(b"not a valid jpeg or png file header")
    r2 = client.post("/api/predict", data={"leaf_image": (corrupt_buf, "bad.jpg")}, content_type="multipart/form-data")
    assert r2.status_code == 400
    assert r2.is_json
    assert "error" in r2.get_json()


def test_oversized_pixel_bomb_protection():
    """Verify decompression bomb protection rejects images exceeding MAX_IMAGE_PIXELS."""
    pytest.importorskip("flask")
    import sys
    sys.path.insert(0, str(ROOT))
    import pi_app

    client = pi_app.app.test_client()

    # Temporarily set MAX_IMAGE_PIXELS to a very small threshold
    orig_max = pi_app.MAX_IMAGE_PIXELS
    pi_app.MAX_IMAGE_PIXELS = 100
    try:
        buf = make_test_image_bytes(size=(20, 20))  # 400 pixels > 100
        response = client.post("/api/predict", data={"leaf_image": (buf, "big.jpg")}, content_type="multipart/form-data")
        assert response.status_code == 400
        assert "error" in response.get_json()
    finally:
        pi_app.MAX_IMAGE_PIXELS = orig_max


def test_unknown_api_endpoint_returns_json_404():
    """API route misses must not fall through to Flask's HTML 404 page."""
    pytest.importorskip("flask")
    import sys
    sys.path.insert(0, str(ROOT))
    import pi_app

    response = pi_app.app.test_client().get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.is_json
    assert response.get_json() == {
        "success": False,
        "error": "Endpoint not found",
    }


# -------------------------------------------------------------
# 5. CONFIGURATION & SCRIPT CONSISTENCY
# -------------------------------------------------------------

def test_requirements_pi_excludes_pytorch_tensorflow():
    """Verify requirements-pi.txt excludes heavyweight PyTorch and TensorFlow dependencies."""
    req_path = ROOT / "requirements-pi.txt"
    assert req_path.is_file()
    lines = [line.strip().lower() for line in req_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    for line in lines:
        assert not line.startswith("torch"), f"PyTorch should not be in requirements-pi: {line}"
        assert not line.startswith("tensorflow"), f"TensorFlow should not be in requirements-pi: {line}"
        assert not line.startswith("keras"), f"Keras should not be in requirements-pi: {line}"

    pkg_names = [line.split(">=")[0].split("<")[0].split("==")[0].strip() for line in lines]
    assert "onnxruntime" in pkg_names
    assert "flask" in pkg_names
    assert "pillow" in pkg_names
    assert "gunicorn" in pkg_names


def test_service_and_setup_script_paths_and_env():
    """Verify setup_and_run_pi.sh and plant-disease-detector.service specify consistent defaults and paths."""
    setup_sh = (ROOT / "setup_and_run_pi.sh").read_text(encoding="utf-8")
    service = (ROOT / "plant-disease-detector.service").read_text(encoding="utf-8")

    # Consistent gunicorn parameters
    assert "5000" in setup_sh and "5000" in service
    assert "workers" in setup_sh and "workers" in service
    assert "threads" in setup_sh and "threads" in service
    assert "timeout" in setup_sh and "timeout" in service
    assert "pi_app:app" in setup_sh and "pi_app:app" in service

    # Check setup script checks treatment_advice.json
    assert "treatment_advice.json" in setup_sh
