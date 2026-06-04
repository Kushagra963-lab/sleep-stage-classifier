from __future__ import annotations

import math

from fastapi.testclient import TestClient

from api.app import app
from api.schemas import STAGE_LABELS

client = TestClient(app)


def sample_payload() -> dict[str, object]:
    signal = [math.sin(index / 12) + 0.2 * math.sin(index / 3) for index in range(600)]
    return {"sample_rate": 100, "channel_name": "Fpz-Cz", "signal": signal}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_stage_probabilities() -> None:
    response = client.post("/predict", json=sample_payload())
    assert response.status_code == 200

    body = response.json()
    assert body["sleep_stage"] in STAGE_LABELS
    assert 0 <= body["class_index"] < len(STAGE_LABELS)
    assert len(body["probabilities"]) == len(STAGE_LABELS)
    assert 0.99 <= sum(item["probability"] for item in body["probabilities"]) <= 1.01


def test_shap_endpoint_returns_regions() -> None:
    response = client.post("/shap", json=sample_payload())
    assert response.status_code == 200

    body = response.json()
    assert body["sleep_stage"] in STAGE_LABELS
    assert len(body["attribution"]) == 12
    assert body["important_regions"]


def test_gradcam_endpoint_returns_heatmap() -> None:
    response = client.post("/gradcam", json=sample_payload())
    assert response.status_code == 200

    body = response.json()
    assert body["sleep_stage"] in STAGE_LABELS
    assert len(body["heatmap"]) == 60
    assert max(body["heatmap"]) <= 1.0

