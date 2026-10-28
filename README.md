# Plant Disease Detector

This project runs a frozen-backbone MobileNetV2 classifier for 38 PlantVillage
classes. Training uses PyTorch and exports an ONNX model; the Raspberry Pi
runtime uses Flask, Pillow, NumPy, and ONNX Runtime on the CPU only.

## Repository

- `train.py` - deterministic dataset discovery, cached feature training,
  evaluation, ONNX export, parity validation, and deployment metadata.
- `pi_app.py` - Flask web UI, `/healthz`, and JSON `POST /predict`.
- `model.onnx`, `class_names.json`, `treatment_advice.json` - deployment
  artifacts that must come from the same training run.
- `setup_and_run_pi.sh` - Raspberry Pi setup and local-network launch.
- `plant-disease-detector.service` - optional systemd unit.

## Training

Use a training machine with Python and the packages in
`requirements-train.txt`:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-train.txt
export PLANT_DATASET_ROOT=/path/to/new-plant-diseases-dataset
python train.py
```

The root must contain sibling `train/` and `valid/` directories with matching
class folders. Automatic Kaggle download is disabled by default; only set
`PLANT_ALLOW_DOWNLOAD=1` when an explicit download is intended. Training
produces the logged metrics, reports, plots, `model.onnx`, and
`deployment_metadata.json`. The script fails if treatment keys or PyTorch/ONNX
probabilities do not match.

## Raspberry Pi

Copy `pi_app.py`, `model.onnx`, `class_names.json`,
`treatment_advice.json`, `requirements-pi.txt`, and `setup_and_run_pi.sh`
to one directory on Raspberry Pi OS 64-bit. Then:

```bash
chmod +x setup_and_run_pi.sh
INSTALL_DEPS=1 ./setup_and_run_pi.sh
```

The script uses one Gunicorn worker and two threads to limit memory use. Open
the printed `http://<pi-ip>:5000` address from a phone on the same Wi-Fi.
This is local HTTP for a trusted network; do not expose it directly to the
public internet.

For boot-time service management, copy the files to `/opt/plant-disease-detector`,
create an unprivileged user, then:

```bash
sudo useradd --system --home /opt/plant-disease-detector plantdetector
sudo chown -R plantdetector:plantdetector /opt/plant-disease-detector
sudo cp plant-disease-detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now plant-disease-detector
systemctl status plant-disease-detector
journalctl -u plant-disease-detector -f
```

## API

```bash
curl -F leaf_image=@leaf.jpg http://<pi-ip>:5000/predict
```

The JSON response contains `predictions`, `top_prediction`, `advice`,
`low_confidence`, and the configured `confidence_threshold`. The threshold is
configurable with `PLANT_MIN_CONFIDENCE` and is a warning only: softmax
confidence is not proof that an image is a supported plant leaf. Set
`PLANT_MAX_IMAGE_PIXELS` to adjust decompression-bomb protection if required.

## Checks and limitations

Run `python -m pytest` for artifact and runtime checks. Training and full
parity validation require the original dataset and training dependencies.
Real Raspberry Pi ARM64 installation, latency, memory use, calibration, and
real-world plant photographs still require validation on that hardware and
data. The model is a screening aid, not a laboratory diagnosis.
