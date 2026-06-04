from __future__ import annotations

from fastapi import FastAPI, HTTPException

from api.schemas import (
    EEGWindowRequest,
    GradCamResponse,
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
    ShapExplanationResponse,
)
from api.services import (
    build_gradcam_response,
    build_health_response,
    build_metadata_response,
    build_prediction_response,
    build_shap_response,
)

app = FastAPI(
    title="Sleep Stage Classifier API",
    description="Classify EEG windows into Wake, N1, N2, N3, or REM sleep stages.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return build_health_response()


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    return build_metadata_response()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: EEGWindowRequest) -> PredictionResponse:
    try:
        return build_prediction_response(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/shap", response_model=ShapExplanationResponse)
def shap(request: EEGWindowRequest) -> ShapExplanationResponse:
    try:
        return build_shap_response(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/gradcam", response_model=GradCamResponse)
def gradcam(request: EEGWindowRequest) -> GradCamResponse:
    try:
        return build_gradcam_response(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

