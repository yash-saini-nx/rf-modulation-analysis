"""Train RF modulation models with PyTorch.

Run after generating data/signals.mat from MATLAB:
    python src/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import prepare_dataset
from model import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "signals.mat"
MODELS_DIR = ROOT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X, y, batch_size=64, shuffle=True):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_classifier(model, train_loader, epochs=20):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.long().to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        print(f"epoch {epoch + 1:02d}/{epochs} loss={total_loss / len(train_loader.dataset):.4f}")


def train_regressor(model, train_loader, epochs=15):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, snr_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            snr_batch = snr_batch.float().to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, snr_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        print(f"snr epoch {epoch + 1:02d}/{epochs} mse={total_loss / len(train_loader.dataset):.4f}")


def predict_classes(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            batch = torch.tensor(X[start:start + 256], dtype=torch.float32).to(DEVICE)
            logits = model(batch)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def predict_regression(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            batch = torch.tensor(X[start:start + 256], dtype=torch.float32).to(DEVICE)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def evaluate_by_snr(model, X_test, y_test, snr_test):
    y_pred = predict_classes(model, X_test)
    rows = []
    for snr in sorted(set(snr_test)):
        mask = snr_test == snr
        rows.append({"snr_db": float(snr), "accuracy": float(accuracy_score(y_test[mask], y_pred[mask]))})
    return pd.DataFrame(rows)


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    ds = prepare_dataset(DATA_PATH)
    num_classes = len(ds.label_encoder.classes_)

    train_loader = make_loader(ds.X_train, ds.y_train)
    summary = {}

    for name in ["cnn_baseline", "cldnn_lite"]:
        print(f"Training {name} on {DEVICE}...")
        model = build_model(name, num_classes)
        train_classifier(model, train_loader)
        torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")

        y_pred = predict_classes(model, ds.X_test)
        acc = accuracy_score(ds.y_test, y_pred)
        evaluate_by_snr(model, ds.X_test, ds.y_test, ds.snr_test).to_csv(
            MODELS_DIR / f"{name}_snr_accuracy.csv", index=False
        )
        np.savetxt(MODELS_DIR / f"{name}_confusion_matrix.csv", confusion_matrix(ds.y_test, y_pred), delimiter=",", fmt="%d")
        summary[name] = {"test_accuracy": float(acc)}

    print("Training snr_regressor...")
    snr_model = build_model("snr_regressor", num_classes)
    snr_loader = make_loader(ds.X_train, ds.snr_train.astype(np.float32))
    train_regressor(snr_model, snr_loader)
    torch.save(snr_model.state_dict(), MODELS_DIR / "snr_regressor.pt")
    snr_pred = predict_regression(snr_model, ds.X_test)
    summary["snr_regressor"] = {"test_mae_db": float(np.mean(np.abs(snr_pred - ds.snr_test)))}

    metadata = {
        "classes": ds.label_encoder.classes_.tolist(),
        "input_shape": list(ds.X_train.shape[1:]),
        "framework": "pytorch",
        "summary": summary,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
