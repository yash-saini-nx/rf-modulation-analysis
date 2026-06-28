"""PyTorch CNN, CLDNN-lite, and SNR regression models."""
from __future__ import annotations

import torch
from torch import nn


class CNNBaseline(nn.Module):
    """Deeper CNN classifier with 4 conv layers for better QAM separation."""

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
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is (batch, 128, 2); Conv1d needs (batch, 2, 128).
        x = x.transpose(1, 2)
        return self.classifier(self.features(x))


class CLDNNLite(nn.Module):
    """CNN + LSTM classifier with residual connection and deeper LSTM."""

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
        # Project original IQ (2 channels) to match conv output (96 channels)
        # after downsampling by 4x (two MaxPool1d(2) layers).
        self.residual_proj = nn.Sequential(
            nn.Conv1d(2, 96, kernel_size=1),
            nn.AvgPool1d(4),
        )
        self.lstm = nn.LSTM(
            input_size=96, hidden_size=96,
            num_layers=2, batch_first=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_channels = x.transpose(1, 2)          # (B, 2, 128)
        conv_out = self.conv(x_channels)         # (B, 96, 32)
        residual = self.residual_proj(x_channels)  # (B, 96, 32)
        fused = conv_out + residual              # residual connection
        fused = fused.transpose(1, 2)            # (B, 32, 96)
        _, (hidden, _) = self.lstm(fused)
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
