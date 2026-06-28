"""Comprehensive model diagnostics: overfitting, underfitting, and accuracy report.

Run from the project root:
    python diagnose_models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import prepare_dataset          # noqa: E402
from model import build_model                # noqa: E402

DATA_PATH = ROOT / "data" / "signals.mat"
MODELS_DIR = ROOT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helpers ────────────────────────────────────────────────────────────
def predict_classes(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            batch = torch.tensor(X[start : start + 256], dtype=torch.float32).to(DEVICE)
            preds.append(torch.argmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(preds)


def predict_regression(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            batch = torch.tensor(X[start : start + 256], dtype=torch.float32).to(DEVICE)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── main ───────────────────────────────────────────────────────────────
def main():
    # Load data with the same split used during training
    ds = prepare_dataset(DATA_PATH)
    classes = ds.label_encoder.classes_
    num_classes = len(classes)

    separator("DATASET OVERVIEW")
    print(f"  Total samples:   {len(ds.X_train) + len(ds.X_test)}")
    print(f"  Training set:    {len(ds.X_train)} ({len(ds.X_train) / (len(ds.X_train) + len(ds.X_test)) * 100:.0f}%)")
    print(f"  Test set:        {len(ds.X_test)} ({len(ds.X_test) / (len(ds.X_train) + len(ds.X_test)) * 100:.0f}%)")
    print(f"  Classes ({num_classes}):    {list(classes)}")
    print(f"  Input shape:     {ds.X_train.shape[1:]}  (128 time-steps × 2 IQ channels)")
    print(f"  SNR range:       {ds.snr_test.min():.0f} dB  to  {ds.snr_test.max():.0f} dB")

    # Class balance check
    separator("CLASS BALANCE CHECK")
    train_counts = pd.Series(ds.y_train).value_counts().sort_index()
    test_counts = pd.Series(ds.y_test).value_counts().sort_index()
    balance_df = pd.DataFrame({
        "Class": classes,
        "Train": train_counts.values,
        "Test": test_counts.values,
        "Total": train_counts.values + test_counts.values,
    })
    print(balance_df.to_string(index=False))
    imbalance = train_counts.max() / train_counts.min()
    if imbalance > 1.5:
        print(f"\n  [!!] CLASS IMBALANCE DETECTED: max/min ratio = {imbalance:.2f}")
    else:
        print(f"\n  [OK] Classes are balanced (max/min ratio = {imbalance:.2f})")

    # ── Per-model diagnostics ──────────────────────────────────────────
    for model_name in ["cnn_baseline", "cldnn_lite"]:
        model_path = MODELS_DIR / f"{model_name}.pt"
        if not model_path.exists():
            print(f"\n  [SKIP] {model_name} — weights file not found")
            continue

        separator(f"MODEL: {model_name.upper()}")

        model = build_model(model_name, num_classes).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {trainable_params:,} trainable / {total_params:,} total")

        # ── 1. Train vs Test accuracy (OVERFITTING CHECK) ─────────────
        y_pred_train = predict_classes(model, ds.X_train)
        y_pred_test = predict_classes(model, ds.X_test)

        train_acc = accuracy_score(ds.y_train, y_pred_train)
        test_acc = accuracy_score(ds.y_test, y_pred_test)
        gap = train_acc - test_acc

        print(f"\n  --- Overfitting Check (Train vs Test) ---")
        print(f"  Train Accuracy:  {train_acc * 100:.2f}%")
        print(f"  Test  Accuracy:  {test_acc * 100:.2f}%")
        print(f"  Gap (train-test): {gap * 100:.2f}%")

        if gap > 0.10:
            print(f"  [!!] OVERFITTING: Train accuracy is {gap*100:.1f}% higher than test.")
            print(f"     The model has memorized training data and generalizes poorly.")
        elif gap > 0.05:
            print(f"  [!!] MILD OVERFITTING: {gap*100:.1f}% gap -- consider adding regularization.")
        elif test_acc < 0.60:
            print(f"  [!!] UNDERFITTING: Test accuracy is only {test_acc*100:.1f}%.")
            print(f"     The model is not learning the patterns well enough.")
        else:
            print(f"  [OK] No significant overfitting (gap < 5%).")

        # ── 2. F1 Score ───────────────────────────────────────────────
        f1_macro = f1_score(ds.y_test, y_pred_test, average="macro")
        f1_weighted = f1_score(ds.y_test, y_pred_test, average="weighted")
        print(f"\n  F1 Score (macro):    {f1_macro:.4f}")
        print(f"  F1 Score (weighted): {f1_weighted:.4f}")

        # ── 3. Per-class accuracy (UNDERFITTING CHECK) ────────────────
        print(f"\n  --- Per-Class Performance (Test Set) ---")
        report = classification_report(
            ds.y_test, y_pred_test, target_names=classes, output_dict=True
        )
        rows = []
        for cls in classes:
            r = report[cls]
            rows.append({
                "Class": cls,
                "Precision": f"{r['precision']:.3f}",
                "Recall": f"{r['recall']:.3f}",
                "F1": f"{r['f1-score']:.3f}",
                "Support": int(r['support']),
            })
        print(pd.DataFrame(rows).to_string(index=False))

        weak_classes = [cls for cls in classes if report[cls]["f1-score"] < 0.65]
        if weak_classes:
            print(f"\n  [!!] WEAK CLASSES (F1 < 0.65): {weak_classes}")
            print(f"     The model struggles to learn these modulations.")
        else:
            print(f"\n  [OK] All classes have F1 >= 0.65")

        # ── 4. Confusion hotspots ─────────────────────────────────────
        print(f"\n  --- Top Confusion Pairs ---")
        cm = confusion_matrix(ds.y_test, y_pred_test)
        # Zero out diagonal and find worst off-diagonal pairs
        cm_off = cm.copy().astype(float)
        np.fill_diagonal(cm_off, 0)
        # Normalize by row totals to get misclassification rates
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_rate = cm_off / np.maximum(row_sums, 1)

        pairs = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and cm_off[i, j] > 0:
                    pairs.append((classes[i], classes[j], cm_off[i, j], cm_rate[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)

        for true_cls, pred_cls, count, rate in pairs[:6]:
            print(f"    {true_cls:>6} -> {pred_cls:<6}  :  {int(count):4d} errors  ({rate * 100:.1f}% of {true_cls} samples)")

        # ── 5. Accuracy by SNR (UNDERFITTING AT LOW SNR) ──────────────
        print(f"\n  --- Accuracy by SNR ---")
        snr_rows = []
        for snr in sorted(set(ds.snr_test)):
            mask = ds.snr_test == snr
            acc = accuracy_score(ds.y_test[mask], y_pred_test[mask])
            n = mask.sum()
            snr_rows.append({"SNR (dB)": f"{snr:+.0f}", "Accuracy": f"{acc*100:.1f}%", "Samples": n})
        print(pd.DataFrame(snr_rows).to_string(index=False))

        low_snr_mask = ds.snr_test <= 0
        high_snr_mask = ds.snr_test >= 10
        if low_snr_mask.sum() > 0 and high_snr_mask.sum() > 0:
            low_acc = accuracy_score(ds.y_test[low_snr_mask], y_pred_test[low_snr_mask])
            high_acc = accuracy_score(ds.y_test[high_snr_mask], y_pred_test[high_snr_mask])
            print(f"\n  Low SNR (<=0 dB)  accuracy: {low_acc*100:.1f}%")
            print(f"  High SNR (>=10 dB) accuracy: {high_acc*100:.1f}%")
            if high_acc < 0.90:
                print(f"  [!!] Even at high SNR the model only reaches {high_acc*100:.1f}%.")
                print(f"     Suggests underfitting or architectural limits.")

    # ── SNR Regressor diagnostics ─────────────────────────────────────
    snr_path = MODELS_DIR / "snr_regressor.pt"
    if snr_path.exists():
        separator("MODEL: SNR_REGRESSOR")
        snr_model = build_model("snr_regressor", num_classes).to(DEVICE)
        snr_model.load_state_dict(torch.load(snr_path, map_location=DEVICE))
        snr_model.eval()

        total_params = sum(p.numel() for p in snr_model.parameters())
        print(f"  Parameters: {total_params:,}")

        snr_pred_train = predict_regression(snr_model, ds.X_train)
        snr_pred_test = predict_regression(snr_model, ds.X_test)

        train_mae = np.mean(np.abs(snr_pred_train - ds.snr_train))
        test_mae = np.mean(np.abs(snr_pred_test - ds.snr_test))
        train_rmse = np.sqrt(np.mean((snr_pred_train - ds.snr_train) ** 2))
        test_rmse = np.sqrt(np.mean((snr_pred_test - ds.snr_test) ** 2))

        print(f"\n  --- Overfitting Check ---")
        print(f"  Train MAE:  {train_mae:.2f} dB")
        print(f"  Test  MAE:  {test_mae:.2f} dB")
        print(f"  Train RMSE: {train_rmse:.2f} dB")
        print(f"  Test  RMSE: {test_rmse:.2f} dB")
        print(f"  MAE Gap:    {abs(test_mae - train_mae):.2f} dB")

        if abs(test_mae - train_mae) > 1.5:
            print(f"  [!!] OVERFITTING on SNR regressor")
        else:
            print(f"  [OK] No significant overfitting on SNR regressor")

        # Per-SNR regression error
        print(f"\n  --- Regression Error by True SNR ---")
        for snr in sorted(set(ds.snr_test)):
            mask = ds.snr_test == snr
            mae = np.mean(np.abs(snr_pred_test[mask] - ds.snr_test[mask]))
            bias = np.mean(snr_pred_test[mask] - ds.snr_test[mask])
            print(f"    SNR {snr:+5.0f} dB  |  MAE: {mae:.2f} dB  |  Bias: {bias:+.2f} dB")

    # ── Summary ────────────────────────────────────────────────────────
    separator("SUMMARY")
    print("  See above for detailed per-model diagnostics.")
    print("  Key things to look for:")
    print("    1. Train-Test accuracy gap  > 10%  →  overfitting")
    print("    2. Test accuracy overall    < 60%  →  underfitting")
    print("    3. Any class with F1        < 0.65 →  class-level underfitting")
    print("    4. High SNR accuracy        < 90%  →  architectural ceiling")
    print("    5. Large confusion pair rates       →  modulation similarity issue")


if __name__ == "__main__":
    main()
