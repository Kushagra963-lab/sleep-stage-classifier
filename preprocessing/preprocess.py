from __future__ import annotations

import numpy as np

try:
    from scipy.signal import butter, filtfilt, welch
except ImportError:  # pragma: no cover - scipy is installed in normal environments.
    butter = None
    filtfilt = None
    welch = None

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
    "gamma": (30.0, 45.0),
}


def as_signal_array(signal: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32).reshape(-1)
    if array.size < 2:
        raise ValueError("EEG signal must contain at least two samples.")
    if not np.isfinite(array).all():
        raise ValueError("EEG signal contains NaN or infinite values.")
    return array


def normalize_signal(signal: list[float] | np.ndarray) -> np.ndarray:
    array = as_signal_array(signal)
    mean = float(np.mean(array))
    std = float(np.std(array))
    if std < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - mean) / std).astype(np.float32)


def bandpass_filter(
    signal: list[float] | np.ndarray,
    sample_rate: float,
    low_hz: float = 0.5,
    high_hz: float = 45.0,
) -> np.ndarray:
    array = as_signal_array(signal)
    if butter is None or filtfilt is None:
        return array

    nyquist = sample_rate / 2.0
    if nyquist <= low_hz:
        return array

    high = min(high_hz, nyquist * 0.95) / nyquist
    low = low_hz / nyquist
    if high <= low:
        return array

    b_coeff, a_coeff = butter(4, [low, high], btype="bandpass")
    pad_length = 3 * max(len(a_coeff), len(b_coeff))
    if array.size <= pad_length:
        return array
    return filtfilt(b_coeff, a_coeff, array).astype(np.float32)


def fit_to_length(signal: list[float] | np.ndarray, target_length: int) -> np.ndarray:
    array = as_signal_array(signal)
    if array.size == target_length:
        return array.astype(np.float32)
    if array.size < target_length:
        return np.pad(array, (0, target_length - array.size), mode="constant").astype(np.float32)

    old_positions = np.linspace(0.0, 1.0, num=array.size, dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(new_positions, old_positions, array).astype(np.float32)


def prepare_model_window(
    signal: list[float] | np.ndarray,
    sample_rate: float,
    target_length: int = 3000,
) -> np.ndarray:
    filtered = bandpass_filter(signal, sample_rate)
    normalized = normalize_signal(filtered)
    return fit_to_length(normalized, target_length)


def epoch_signal(
    signal: list[float] | np.ndarray,
    sample_rate: float,
    epoch_seconds: int = 30,
) -> np.ndarray:
    array = as_signal_array(signal)
    samples_per_epoch = int(sample_rate * epoch_seconds)
    if samples_per_epoch <= 0:
        raise ValueError("Epoch length must contain at least one sample.")
    epoch_count = array.size // samples_per_epoch
    if epoch_count == 0:
        return np.empty((0, samples_per_epoch), dtype=np.float32)
    trimmed = array[: epoch_count * samples_per_epoch]
    return trimmed.reshape(epoch_count, samples_per_epoch).astype(np.float32)


def band_power_features(signal: list[float] | np.ndarray, sample_rate: float) -> dict[str, float]:
    array = as_signal_array(signal)
    if welch is None:
        frequencies = np.fft.rfftfreq(array.size, d=1.0 / sample_rate)
        density = np.abs(np.fft.rfft(array)) ** 2
        return _integrate_band_powers(frequencies, density, array)

    segment_length = min(256, array.size)
    frequencies, density = welch(array, fs=sample_rate, nperseg=segment_length)
    return _integrate_band_powers(frequencies, density, array)


def _integrate_band_powers(
    frequencies: np.ndarray,
    density: np.ndarray,
    signal: np.ndarray,
) -> dict[str, float]:
    powers = {}
    total_power = 1e-12

    raw_band_powers: dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        power = float(np.trapz(density[mask], frequencies[mask])) if np.any(mask) else 0.0
        raw_band_powers[name] = max(power, 0.0)
        total_power += max(power, 0.0)

    for name, power in raw_band_powers.items():
        powers[name] = power / total_power
    powers["variance"] = float(np.var(signal))
    return powers
