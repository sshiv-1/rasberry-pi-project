#!/usr/bin/env bash
set -euo pipefail

# Run once on Raspberry Pi OS 64-bit from the project directory.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed. On Raspberry Pi OS run: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
  exit 1
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "WARNING: this device reports architecture $(uname -m). ONNX Runtime wheels are most straightforward on 64-bit Raspberry Pi OS (aarch64)."
fi

if command -v apt-get >/dev/null 2>&1; then
  echo "Installing OS prerequisites (sudo may ask for your password)..."
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-pip libgomp1
fi

if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  echo "Creating Python virtual environment and installing application dependencies..."
  python3 -m venv "$APP_DIR/.venv"
fi

if [[ "${INSTALL_DEPS:-0}" == "1" || ! -x "$APP_DIR/.venv/bin/gunicorn" ]]; then
  "$APP_DIR/.venv/bin/python" -m pip install --upgrade pip
  "$APP_DIR/.venv/bin/python" -m pip install -r "$APP_DIR/requirements-pi.txt"
fi

# Give a clear message before starting if model assets were not copied into this folder.
for file in model.onnx class_names.json treatment_advice.json; do
  if [[ ! -f "$APP_DIR/$file" ]]; then
    echo "ERROR: $file is missing. Copy it into the same folder as pi_app.py (exported by train.py)."
    exit 1
  fi
done

if [[ ! -f "$APP_DIR/pi_app.py" ]]; then
  echo "ERROR: pi_app.py is missing."
  exit 1
fi

"$APP_DIR/.venv/bin/python" - <<'PY'
from pathlib import Path
import flask, numpy, onnxruntime, PIL
session = onnxruntime.InferenceSession(
    str(Path("model.onnx")),
    providers=["CPUExecutionProvider"],
)
print("Runtime packages OK: Flask", flask.__version__, "| NumPy", numpy.__version__, "| ONNX Runtime", onnxruntime.__version__, "| Pillow", PIL.__version__)
print("ONNX model OK:", session.get_inputs()[0].shape, "->", session.get_outputs()[0].shape)
PY

echo
echo "Starting local-network app. Press Ctrl+C to stop it."
echo "From another device on the same Wi-Fi, open: http://$(hostname -I | awk '{print $1}'):5000"
echo "This is intended for trusted local networks, not direct public internet exposure."
exec "$APP_DIR/.venv/bin/gunicorn" \
  --bind "0.0.0.0:${PORT:-5000}" \
  --workers "${GUNICORN_WORKERS:-1}" \
  --threads "${GUNICORN_THREADS:-2}" \
  --timeout "${GUNICORN_TIMEOUT:-120}" \
  pi_app:app
