"""PyTorch CNN, CLDNN-lite, and SNR regression models."""
from __future__ import annotations

import torch
from torch import nn


class CNNBaseline(nn.Module):
    """Simple CNN-only classifier used as the baseline model."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is (batch, 128, 2); Conv1d needs (batch, 2, 128).
        x = x.transpose(1, 2)
        return self.classifier(self.features(x))


class CLDNNLite(nn.Module):
    """CNN + LSTM classifier inspired by RF modulation CLDNN models."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=96, hidden_size=64, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


class SNRRegressor(nn.Module):
    """Small regression model that estimates SNR in dB from IQ samples."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=48, batch_first=True)
        self.regressor = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        return self.regressor(hidden[-1]).squeeze(-1)


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "cnn_baseline":
        return CNNBaseline(num_classes)
    if name == "cldnn_lite":
        return CLDNNLite(num_classes)
    if name == "snr_regressor":
        return SNRRegressor()
    raise ValueError(f"Unknown model name: {name}")
