"""Dataset utilities for RF modulation analysis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    snr_train: np.ndarray
    snr_val: np.ndarray
    snr_test: np.ndarray
    label_encoder: LabelEncoder


def load_mat_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load MATLAB-generated IQ data.

    Expected shape from MATLAB is (samples, 2, 128). PyTorch training code uses
    (samples, time_steps, channels), so we transpose to (samples, 128, 2).

    MATLAB ``string`` arrays are saved as MCOS objects that scipy cannot decode.
    When that happens we reconstruct labels from ``modTypes`` and ``snrValues``
    using the known generation order (mod x snr x samplesPerClassPerSnr).
    """
    data = loadmat(path)
    X = np.asarray(data["iqData"], dtype=np.float32)
    snrs = np.asarray(data["snrs"], dtype=np.float32).reshape(-1)

    # Try to read labels directly; fall back to reconstruction if they are
    # stored as an opaque MATLAB string (MCOS) object.
    try:
        labels = np.asarray(data["labels"]).astype(str).reshape(-1)
    except (TypeError, ValueError):
        # Reconstruct from modTypes and snrValues stored alongside the data.
        mod_types = [a.item() for a in data["modTypes"].reshape(-1)]
        snr_values = data["snrValues"].reshape(-1).tolist()
        n_total = X.shape[0]
        samples_per_class_per_snr = n_total // (len(mod_types) * len(snr_values))
        labels = np.array(
            [
                mod
                for mod in mod_types
                for _snr in snr_values
                for _ in range(samples_per_class_per_snr)
            ]
        )

    if X.shape[1] == 2:
        X = np.transpose(X, (0, 2, 1))

    return X, labels, snrs


def normalize_iq(X: np.ndarray) -> np.ndarray:
    """Normalize each IQ example independently to keep values stable."""
    max_abs = np.max(np.abs(X), axis=(1, 2), keepdims=True)
    return X / np.maximum(max_abs, 1e-8)


def prepare_dataset(
    path: str | Path,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> DatasetBundle:
    """Load data and split into train / validation / test sets.

    The test set is carved out first (``test_size`` of the total data).
    Then the remaining data is split into train and validation sets
    (``val_size`` of the remaining data).  With the defaults this gives
    a 64 / 16 / 20 split.
    """
    X, labels, snrs = load_mat_dataset(path)
    X = normalize_iq(X)

    encoder = LabelEncoder()
    y_idx = encoder.fit_transform(labels).astype(np.int64)

    # First split: carve out the test set.
    X_dev, X_test, y_dev, y_test, snr_dev, snr_test = train_test_split(
        X, y_idx, snrs, test_size=test_size, random_state=seed, stratify=y_idx
    )

    # Second split: carve out the validation set from the remaining data.
    X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
        X_dev, y_dev, snr_dev, test_size=val_size, random_state=seed, stratify=y_dev
    )

    return DatasetBundle(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        snr_train, snr_val, snr_test,
        encoder,
    )
