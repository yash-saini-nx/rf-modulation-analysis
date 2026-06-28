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


def evaluate_loader(model, loader, loss_fn):
    """Compute average loss and accuracy over a DataLoader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.long().to(DEVICE)
            logits = model(X_batch)
            total_loss += loss_fn(logits, y_batch).item() * len(X_batch)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += len(X_batch)
    return total_loss / total, correct / total


def train_classifier(model, train_loader, val_loader, name, epochs=40, patience=5):
    """Train classifier with validation tracking, early stopping, and LR scheduling."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.long().to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            train_total += len(X_batch)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validation ---
        val_loss, val_acc = evaluate_loader(model, val_loader, loss_fn)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  epoch {epoch + 1:02d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"lr={current_lr:.1e}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 4),
            "lr": current_lr,
        })

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    # Save learning curves
    pd.DataFrame(history).to_csv(MODELS_DIR / f"{name}_learning_curves.csv", index=False)
    return model


def train_regressor(model, train_loader, val_loader, name, epochs=30, patience=5):
    """Train SNR regressor with validation tracking and early stopping."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0.0
        total_n = 0
        for X_batch, snr_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            snr_batch = snr_batch.float().to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, snr_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
            total_n += len(X_batch)

        train_mse = total_loss / total_n

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for X_batch, snr_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                snr_batch = snr_batch.float().to(DEVICE)
                pred = model(X_batch)
                val_loss += loss_fn(pred, snr_batch).item() * len(X_batch)
                val_n += len(X_batch)
        val_mse = val_loss / val_n

        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  snr epoch {epoch + 1:02d}/{epochs}  "
            f"train_mse={train_mse:.4f}  val_mse={val_mse:.4f}  "
            f"lr={current_lr:.1e}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_mse, 5),
            "val_loss": round(val_mse, 5),
            "lr": current_lr,
        })

        # --- Early Stopping ---
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    pd.DataFrame(history).to_csv(MODELS_DIR / f"{name}_learning_curves.csv", index=False)
    return model


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

    print(f"Dataset: {len(ds.X_train)} train / {len(ds.X_val)} val / {len(ds.X_test)} test")
    print(f"Classes ({num_classes}): {list(ds.label_encoder.classes_)}")
    print(f"Device: {DEVICE}\n")

    train_loader = make_loader(ds.X_train, ds.y_train)
    val_loader = make_loader(ds.X_val, ds.y_val, shuffle=False)
    summary = {}

    for name in ["cnn_baseline", "cldnn_lite"]:
        print(f"--- Skipping {name} (already trained) ---")
        model = build_model(name, num_classes)
        model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location=DEVICE))
        
        y_pred = predict_classes(model, ds.X_test)
        acc = accuracy_score(ds.y_test, y_pred)
        evaluate_by_snr(model, ds.X_test, ds.y_test, ds.snr_test).to_csv(
            MODELS_DIR / f"{name}_snr_accuracy.csv", index=False
        )
        np.savetxt(MODELS_DIR / f"{name}_confusion_matrix.csv", confusion_matrix(ds.y_test, y_pred), delimiter=",", fmt="%d")
        summary[name] = {"test_accuracy": float(acc)}
        print(f"  => Test accuracy: {acc:.4f}\n")

    print("--- Training snr_regressor ---")
    snr_model = build_model("snr_regressor", num_classes)
    snr_train_loader = make_loader(ds.X_train, ds.snr_train.astype(np.float32))
    snr_val_loader = make_loader(ds.X_val, ds.snr_val.astype(np.float32), shuffle=False)
    snr_model = train_regressor(snr_model, snr_train_loader, snr_val_loader, "snr_regressor")
    torch.save(snr_model.state_dict(), MODELS_DIR / "snr_regressor.pt")
    snr_pred = predict_regression(snr_model, ds.X_test)
    summary["snr_regressor"] = {"test_mae_db": float(np.mean(np.abs(snr_pred - ds.snr_test)))}
    print(f"  => Test MAE: {summary['snr_regressor']['test_mae_db']:.2f} dB\n")

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
