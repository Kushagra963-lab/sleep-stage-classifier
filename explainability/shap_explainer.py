from __future__ import annotations

from typing import Protocol

import numpy as np

from preprocessing.preprocess import as_signal_array, prepare_model_window


class WindowPredictor(Protocol):
    def predict_prepared_window(self, window: np.ndarray, sample_rate: float) -> np.ndarray:
        """Return class probabilities for a normalized model window."""


def _top_regions(scores: np.ndarray, duration_seconds: float, top_n: int = 3) -> list[dict[str, float]]:
    if scores.size == 0:
        return []

    order = np.argsort(scores)[::-1][:top_n]
    segment_seconds = duration_seconds / scores.size
    regions = []
    for index in order:
        start = float(index * segment_seconds)
        end = float((index + 1) * segment_seconds)
        regions.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "score": round(float(scores[index]), 6),
            }
        )
    return regions


def compute_occlusion_attributions(
    signal: list[float] | np.ndarray,
    sample_rate: float,
    predictor: WindowPredictor,
    segments: int = 12,
) -> dict[str, object]:
    raw_signal = as_signal_array(signal)
    window = prepare_model_window(raw_signal, sample_rate)
    baseline = predictor.predict_prepared_window(window, sample_rate)
    target_index = int(np.argmax(baseline))
    target_probability = float(baseline[target_index])

    edges = np.linspace(0, window.size, num=segments + 1, dtype=int)
    contributions = np.zeros(segments, dtype=np.float64)

    for index in range(segments):
        start, end = int(edges[index]), int(edges[index + 1])
        if start == end:
            continue
        occluded = window.copy()
        occluded[start:end] = 0.0
        occluded_probability = predictor.predict_prepared_window(occluded, sample_rate)[target_index]
        contributions[index] = max(0.0, target_probability - float(occluded_probability))

    if float(contributions.sum()) <= 1e-12:
        contributions = np.array(
            [
                float(np.mean(np.abs(window[int(edges[i]) : int(edges[i + 1])])))
                for i in range(segments)
            ],
            dtype=np.float64,
        )

    if float(contributions.max()) > 0.0:
        contributions = contributions / float(contributions.max())

    duration_seconds = raw_signal.size / sample_rate
    return {
        "target_index": target_index,
        "attribution": [round(float(value), 6) for value in contributions],
        "regions": _top_regions(contributions, duration_seconds),
    }

