from __future__ import annotations

import numpy as np
import pytest

from preprocessing.preprocess import (
    band_power_features,
    epoch_signal,
    normalize_signal,
    prepare_model_window,
)


def test_prepare_model_window_normalizes_and_resizes() -> None:
    signal = np.sin(np.linspace(0, 20, num=500))
    window = prepare_model_window(signal, sample_rate=100)

    assert window.shape == (3000,)
    assert np.isfinite(window).all()
    assert abs(float(np.mean(window))) < 0.25


def test_epoch_signal_splits_complete_windows() -> None:
    signal = np.arange(6500)
    epochs = epoch_signal(signal, sample_rate=100, epoch_seconds=30)

    assert epochs.shape == (2, 3000)
    assert epochs[1, 0] == 3000


def test_band_power_features_are_normalized() -> None:
    signal = np.sin(np.linspace(0, 80, num=3000))
    features = band_power_features(signal, sample_rate=100)
    band_total = sum(features[name] for name in ("delta", "theta", "alpha", "sigma", "beta", "gamma"))

    assert 0.99 <= band_total <= 1.01
    assert features["variance"] >= 0


def test_invalid_signal_raises_value_error() -> None:
    with pytest.raises(ValueError):
        normalize_signal([1.0, float("nan")])

