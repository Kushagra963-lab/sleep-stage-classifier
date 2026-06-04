# Sleep Stage Classifier API

FastAPI service for classifying 30-second EEG windows into sleep stages with CNN-LSTM training hooks and explainability endpoints for signal attributions.

The API runs out of the box with a deterministic demo predictor, so the project can be tested before a trained model is added. For production use, train on Sleep-EDF or another labeled EEG dataset and save the model to `models/sleep_model.keras`.

## Features

- EEG preprocessing: bandpass filtering, normalization, epoching, and band-power extraction
- CNN-LSTM model builder for sleep stage classification
- REST API endpoints for prediction, SHAP-style attributions, and Grad-CAM-style heatmaps
- Dockerfile for API deployment
- GitHub Actions for tests and publishing a Docker image to GitHub Container Registry

## Project Structure

```text
sleep-stage-classifier/
├── api/
│   ├── app.py
│   ├── schemas.py
│   └── services.py
├── data/
├── explainability/
│   ├── gradcam.py
│   └── shap_explainer.py
├── models/
├── notebooks/
├── preprocessing/
│   └── preprocess.py
├── training/
│   ├── model.py
│   └── train.py
├── tests/
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── requirements-ml.txt
└── pyproject.toml
```

## Local Setup

Use Python 3.11 or 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Run the API:

```bash
uvicorn api.app:app --reload
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## API Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sample_rate": 100, "signal": [0.1, 0.2, 0.15, -0.1, 0.0, 0.08]}'
```

Core endpoints:

- `POST /predict` returns the predicted stage and class probabilities.
- `POST /shap` returns signal regions that most influenced the prediction.
- `POST /gradcam` returns a normalized heatmap across the EEG window.
- `GET /metadata` returns stage labels and model details.

## Training

Install ML dependencies when you are ready to train:

```bash
pip install -r requirements-ml.txt
```

Prepare an `.npz` file with:

- `X_train`, `y_train`
- `X_val`, `y_val`
- optionally `X_test`, `y_test`

Expected `X` shape is `(samples, 3000, 1)` for 30-second windows at 100 Hz. Labels can be integer class IDs from `0` to `4`.

```bash
python -m training.train \
  --dataset data/sleep_edf_windows.npz \
  --output models/sleep_model.keras \
  --epochs 50 \
  --batch-size 32
```

Stage mapping:

| Class | Stage |
| --- | --- |
| 0 | Wake |
| 1 | N1 |
| 2 | N2 |
| 3 | N3 |
| 4 | REM |

## Docker

```bash
docker build -t sleep-stage-classifier .
docker run -p 8000:8000 sleep-stage-classifier
```

## GitHub Deployment

This repo includes `.github/workflows/docker.yml`. After pushing to GitHub, every push to `main` builds and publishes the API Docker image to GitHub Container Registry:

```text
ghcr.io/<owner>/<repo>:latest
```

The image can then be deployed to any container host that supports GHCR images.

## Notes

- The default predictor is a demo baseline so tests and API demos work without a trained model.
- To use a trained CNN-LSTM, place the Keras model at `models/sleep_model.keras`.
- SHAP and Grad-CAM endpoints use lightweight signal attribution fallbacks unless optional ML dependencies and a compatible trained model are available.

