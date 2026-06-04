from __future__ import annotations

import numpy as np

from preprocessing.preprocess import as_signal_array, prepare_model_window


def _top_regions(scores: np.ndarray, duration_seconds: float, top_n: int = 3) -> list[dict[str, float]]:
    order = np.argsort(scores)[::-1][:top_n]
    segment_seconds = duration_seconds / scores.size
    return [
        {
            "start_seconds": round(float(index * segment_seconds), 3),
            "end_seconds": round(float((index + 1) * segment_seconds), 3),
            "score": round(float(scores[index]), 6),
        }
        for index in order
    ]


def compute_gradcam_heatmap(
    signal: list[float] | np.ndarray,
    sample_rate: float,
    segments: int = 60,
) -> dict[str, object]:
    raw_signal = as_signal_array(signal)
    window = prepare_model_window(raw_signal, sample_rate)

    gradient_energy = np.abs(np.gradient(window))
    amplitude_energy = np.abs(window)
    saliency = (0.65 * amplitude_energy) + (0.35 * gradient_energy)

    edges = np.linspace(0, saliency.size, num=segments + 1, dtype=int)
    heatmap = np.array(
        [float(np.mean(saliency[int(edges[i]) : int(edges[i + 1])])) for i in range(segments)],
        dtype=np.float64,
    )

    if float(heatmap.max()) > 0.0:
        heatmap = heatmap / float(heatmap.max())

    duration_seconds = raw_signal.size / sample_rate
    return {
        "heatmap": [round(float(value), 6) for value in heatmap],
        "regions": _top_regions(heatmap, duration_seconds),
    }
