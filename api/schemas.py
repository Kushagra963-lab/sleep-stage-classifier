from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STAGE_LABELS: tuple[str, ...] = ("Wake", "N1", "N2", "N3", "REM")
StageName = Literal["Wake", "N1", "N2", "N3", "REM"]


class EEGWindowRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sample_rate": 100,
                "channel_name": "Fpz-Cz",
                "signal": [0.1, 0.2, 0.15, -0.1, 0.0, 0.08],
            }
        }
    )

    signal: list[float] = Field(..., min_length=2)
    sample_rate: float = Field(default=100.0, gt=0, le=2048)
    channel_name: str | None = Field(default=None, max_length=80)


class StageProbability(BaseModel):
    stage: str
    probability: float


class PredictionResponse(BaseModel):
    sleep_stage: str
    class_index: int
    confidence: float
    probabilities: list[StageProbability]
    model_version: str
    window_seconds: float


class ImportantRegion(BaseModel):
    start_seconds: float
    end_seconds: float
    score: float


class ShapExplanationResponse(BaseModel):
    sleep_stage: str
    method: str
    important_regions: list[ImportantRegion]
    attribution: list[float]


class GradCamResponse(BaseModel):
    sleep_stage: str
    method: str
    heatmap: list[float]
    important_regions: list[ImportantRegion]


class MetadataResponse(BaseModel):
    service: str
    stage_labels: list[str]
    model_version: str
    model_path: str


class HealthResponse(BaseModel):
    status: str
    model_version: str

