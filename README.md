# LeafLens Plant Disease Detector

LeafLens is a 38-class PlantVillage classifier. The laptop frontend is an
upload-only Streamlit app; inference and treatment advice remain on the
Raspberry Pi; training and ONNX export are kept separate.

## Repository layout

```text
LeafLens/
├── frontend/
│   ├── streamlit_app.py
│   ├── streamlit_api.py
│   └── requirements-streamlit.txt
├── pi/
│   ├── pi_app.py
│   ├── model.onnx
│   ├── class_names.json
│   ├── treatment_advice.json
│   ├── deployment_metadata.json
│   ├── requirements-pi.txt
│   ├── plant-disease-detector.service
│   └── setup_and_run_pi.sh
├── training/
│   ├── train.py
│   └── requirements-train.txt
├── tests/
│   ├── test_artifacts.py
│   ├── test_streamlit_api.py
│   └── fixtures/
├── .streamlit/config.toml
├── .gitignore
└── README.md
```

The complete `pi/` directory is the deployment bundle. It contains the Flask
application, ONNX model, exact class ordering, treatment advice, metadata and
Pi dependencies; it does not require the frontend or training directory.

## Windows frontend

From the repository root, create or activate a virtual environment and install
the frontend dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r frontend\requirements-streamlit.txt
python -m streamlit run frontend\streamlit_app.py
```

Set `PI_API_URL` before launching, or enter the Raspberry Pi base URL in the
sidebar:

```powershell
$env:PI_API_URL = "http://192.168.1.50:5000"
python -m streamlit run frontend\streamlit_app.py
```

The frontend converts uploads to JPEG in memory, checks `/healthz`, and sends
the image to `/api/predict`. It does not load the ONNX model or contain the
treatment dictionary.

## Raspberry Pi inference server

Copy the entire `pi/` directory to Raspberry Pi OS 64-bit. From inside that
directory:

```bash
chmod +x setup_and_run_pi.sh
INSTALL_DEPS=1 ./setup_and_run_pi.sh
```

The script creates `pi/.venv`, installs `requirements-pi.txt`, validates the
ONNX model and starts Gunicorn on port 5000. The API can be called with:

```bash
curl -F leaf_image=@leaf.jpg http://<pi-ip>:5000/api/predict
```

For boot-time service management after copying the directory to
`/opt/plant-disease-detector`:

```bash
sudo useradd --system --home /opt/plant-disease-detector plantdetector
sudo chown -R plantdetector:plantdetector /opt/plant-disease-detector
sudo cp /opt/plant-disease-detector/plant-disease-detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now plant-disease-detector
systemctl status plant-disease-detector
journalctl -u plant-disease-detector -f
```

Do not expose the Flask/Gunicorn service directly to the public internet.

## Training and retraining

Training requires the packages in `training/requirements-train.txt` and a
dataset containing sibling `train/` and `valid/` directories with matching
class folders:

```bash
cd training
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-train.txt
export PLANT_DATASET_ROOT=/path/to/new-plant-diseases-dataset
python train.py
```

`train.py` preserves the 38-class ordering, exports `model.onnx`, writes the
JSON deployment artifacts and performs ONNX parity checks. Copy the resulting
deployment artifacts into `pi/` only after reviewing the generated metadata.
Do not retrain as part of routine cleanup.

## Validation

```powershell
python -m pytest tests\test_artifacts.py -v
python -m pytest tests\test_streamlit_api.py -v
python -m pytest -v
```

The test suite verifies artifact hashes and shapes, CPU ONNX execution, Flask
API behavior, frontend request contracts, and Pi script consistency.
