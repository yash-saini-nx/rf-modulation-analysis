"""DSP preprocessing and plotting helpers."""
from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram


def iq_to_complex(iq: np.ndarray) -> np.ndarray:
    """Convert one IQ sample from shape (128, 2) to complex vector."""
    return iq[:, 0] + 1j * iq[:, 1]


def amplitude_phase(iq: np.ndarray) -> np.ndarray:
    """Represent IQ as amplitude and phase instead of I and Q."""
    z = iq_to_complex(iq)
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))
    return np.stack([amp, phase], axis=-1)


def fft_spectrum(iq: np.ndarray, sample_rate: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency axis and normalized FFT magnitude."""
    z = iq_to_complex(iq)
    spectrum = np.fft.fftshift(np.fft.fft(z))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(z), d=1 / sample_rate))
    magnitude = np.abs(spectrum)
    magnitude = magnitude / np.maximum(np.max(magnitude), 1e-8)
    return freqs, magnitude


def make_spectrogram(iq: np.ndarray, sample_rate: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return spectrogram frequency, time, and power arrays.

    With only 128 IQ samples the raw STFT grid is very coarse.
    Zero-padding the FFT (nfft=128) gives 4× finer frequency resolution,
    and higher overlap produces more time slices.  The power is converted
    to dB scale for perceptually meaningful color mapping.
    """
    z = iq_to_complex(iq)
    freqs, times, power = spectrogram(
        z, fs=sample_rate, nperseg=32, noverlap=28, nfft=128, mode="magnitude",
    )
    # Shift to −fs/2 … +fs/2 for complex signals.
    freqs = np.fft.fftshift(freqs)
    power = np.fft.fftshift(power, axes=0)
    # Convert to dB scale (floor at −40 dB to avoid log(0)).
    power_db = 20 * np.log10(np.maximum(power, 1e-8))
    power_db = np.maximum(power_db, power_db.max() - 40)
    return freqs, times, power_db


def confidence_label(probabilities: np.ndarray, classes: np.ndarray, threshold: float = 0.60) -> tuple[str, float]:
    """Return predicted label unless confidence is too low."""
    idx = int(np.argmax(probabilities))
    confidence = float(probabilities[idx])
    if confidence < threshold:
        return "unclassifiable", confidence
    return str(classes[idx]), confidence
