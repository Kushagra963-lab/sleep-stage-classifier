from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from api.schemas import (
    STAGE_LABELS,
    EEGWindowRequest,
    GradCamResponse,
    HealthResponse,
    ImportantRegion,
    MetadataResponse,
    PredictionResponse,
    ShapExplanationResponse,
    StageProbability,
)
from explainability.gradcam import compute_gradcam_heatmap
from explainability.shap_explainer import compute_occlusion_attributions
from preprocessing.preprocess import as_signal_array, band_power_features, prepare_model_window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sleep_model.keras"
DEMO_MODEL_VERSION = "demo-heuristic-v1"


@dataclass(frozen=True)
class PredictionResult:
    stage_index: int
    probabilities: np.ndarray
    model_version: str
    window_seconds: float

    @property
    def stage(self) -> str:
        return STAGE_LABELS[self.stage_index]

    @property
    def confidence(self) -> float:
        return float(self.probabilities[self.stage_index])


class SleepStagePredictor:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self._keras_model = None
        self._model_load_attempted = False

    @property
    def model_version(self) -> str:
        if self.model_path.exists():
            return f"keras:{self.model_path.name}"
        return DEMO_MODEL_VERSION

    def predict_signal(self, signal: list[float] | np.ndarray, sample_rate: float) -> PredictionResult:
        raw_signal = as_signal_array(signal)
        window = prepare_model_window(raw_signal, sample_rate)
        probabilities = self.predict_prepared_window(window, sample_rate)
        stage_index = int(np.argmax(probabilities))
        return PredictionResult(
            stage_index=stage_index,
            probabilities=probabilities,
            model_version=self.model_version,
            window_seconds=round(raw_signal.size / sample_rate, 3),
        )

    def predict_prepared_window(self, window: np.ndarray, sample_rate: float) -> np.ndarray:
        model = self._load_keras_model()
        if model is not None:
            prediction = model.predict(window.reshape(1, -1, 1), verbose=0)[0]
            return self._clean_probabilities(prediction)
        return self._heuristic_probabilities(window, sample_rate)

    def _load_keras_model(self):
        if self._model_load_attempted:
            return self._keras_model

        self._model_load_attempted = True
        if not self.model_path.exists():
            return None

        try:
            from tensorflow.keras.models import load_model

            self._keras_model = load_model(self.model_path)
        except Exception:
            self._keras_model = None
        return self._keras_model

    @staticmethod
    def _clean_probabilities(values: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
        if probabilities.size != len(STAGE_LABELS):
            raise ValueError("Model returned an unexpected number of class probabilities.")
        probabilities = np.clip(probabilities, 0.0, None)
        total = float(probabilities.sum())
        if total <= 0.0:
            raise ValueError("Model returned invalid probabilities.")
        return probabilities / total

    @staticmethod
    def _heuristic_probabilities(window: np.ndarray, sample_rate: float) -> np.ndarray:
        features = band_power_features(window, sample_rate)
        delta = features["delta"]
        theta = features["theta"]
        alpha = features["alpha"]
        sigma = features["sigma"]
        beta = features["beta"]
        gamma = features["gamma"]
        variance = min(features["variance"], 4.0) / 4.0

        logits = np.array(
            [
                (1.35 * alpha) + (0.55 * beta) + (0.15 * variance),
                (1.30 * theta) + (0.25 * alpha),
                (1.45 * sigma) + (0.50 * theta) + 0.18,
                (2.20 * delta) + (0.20 * sigma),
                (0.55 * theta) + (0.60 * beta) + (0.30 * gamma) + 0.08,
            ],
            dtype=np.float64,
        )
        logits = logits - float(np.max(logits))
        exp_logits = np.exp(logits)
        return exp_logits / float(exp_logits.sum())


predictor = SleepStagePredictor()


def _probability_rows(probabilities: np.ndarray) -> list[StageProbability]:
    return [
        StageProbability(stage=stage, probability=round(float(probabilities[index]), 6))
        for index, stage in enumerate(STAGE_LABELS)
    ]


def build_prediction_response(request: EEGWindowRequest) -> PredictionResponse:
    prediction = predictor.predict_signal(request.signal, request.sample_rate)
    return PredictionResponse(
        sleep_stage=prediction.stage,
        class_index=prediction.stage_index,
        confidence=round(prediction.confidence, 6),
        probabilities=_probability_rows(prediction.probabilities),
        model_version=prediction.model_version,
        window_seconds=prediction.window_seconds,
    )


def build_shap_response(request: EEGWindowRequest) -> ShapExplanationResponse:
    prediction = predictor.predict_signal(request.signal, request.sample_rate)
    explanation = compute_occlusion_attributions(request.signal, request.sample_rate, predictor)
    return ShapExplanationResponse(
        sleep_stage=prediction.stage,
        method="occlusion SHAP-style attribution",
        important_regions=[ImportantRegion(**region) for region in explanation["regions"]],
        attribution=explanation["attribution"],
    )


def build_gradcam_response(request: EEGWindowRequest) -> GradCamResponse:
    prediction = predictor.predict_signal(request.signal, request.sample_rate)
    explanation = compute_gradcam_heatmap(request.signal, request.sample_rate)
    return GradCamResponse(
        sleep_stage=prediction.stage,
        method="signal saliency Grad-CAM-style heatmap",
        heatmap=explanation["heatmap"],
        important_regions=[ImportantRegion(**region) for region in explanation["regions"]],
    )


def build_metadata_response() -> MetadataResponse:
    return MetadataResponse(
        service="sleep-stage-classifier-api",
        stage_labels=list(STAGE_LABELS),
        model_version=predictor.model_version,
        model_path=str(DEFAULT_MODEL_PATH),
    )


def build_health_response() -> HealthResponse:
    return HealthResponse(status="ok", model_version=predictor.model_version)

