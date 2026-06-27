"""Streamlit dashboard for RF Modulation Analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy.io import loadmat

from src.model import build_model
from src.preprocess import confidence_label, fft_spectrum, make_spectrogram

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "signals.mat"
MODELS_DIR = ROOT / "models"
LOG_PATH = ROOT / "models" / "prediction_log.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="RF Modulation Analysis", layout="wide")
st.title("RF Modulation Analysis")
st.caption("MATLAB signal generation + Python DSP + PyTorch CNN/CLDNN classification")


@st.cache_data
def load_signals():
    data = loadmat(DATA_PATH)
    X = np.asarray(data["iqData"], dtype=np.float32)
    snrs = np.asarray(data["snrs"], dtype=np.float32).reshape(-1)
    # MATLAB string arrays are MCOS objects; fall back to reconstruction.
    try:
        labels = np.asarray(data["labels"]).astype(str).reshape(-1)
    except (TypeError, ValueError):
        mod_types = [a.item() for a in data["modTypes"].reshape(-1)]
        snr_values = data["snrValues"].reshape(-1).tolist()
        n_total = X.shape[0]
        spc = n_total // (len(mod_types) * len(snr_values))
        labels = np.array(
            [mod for mod in mod_types for _s in snr_values for _ in range(spc)]
        )
    if X.shape[1] == 2:
        X = np.transpose(X, (0, 2, 1))
    max_abs = np.max(np.abs(X), axis=(1, 2), keepdims=True)
    X = X / np.maximum(max_abs, 1e-8)
    return X, labels, snrs


@st.cache_resource
def load_trained_models():
    metadata_path = MODELS_DIR / "metadata.json"
    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    num_classes = len(metadata["classes"])

    models = {"metadata": metadata}
    for display_name, file_stem in [("CNN Baseline", "cnn_baseline"), ("CLDNN Lite", "cldnn_lite")]:
        model_path = MODELS_DIR / f"{file_stem}.pt"
        if not model_path.exists():
            return None
        model = build_model(file_stem, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        models[display_name] = model

    snr_path = MODELS_DIR / "snr_regressor.pt"
    if not snr_path.exists():
        return None
    snr_model = build_model("snr_regressor", num_classes)
    snr_model.load_state_dict(torch.load(snr_path, map_location=DEVICE))
    snr_model.to(DEVICE)
    snr_model.eval()
    models["SNR Regressor"] = snr_model
    return models


def predict_probabilities(model, iq):
    with torch.no_grad():
        x = torch.tensor(iq[np.newaxis, ...], dtype=torch.float32).to(DEVICE)
        logits = model(x)
        return torch.softmax(logits, dim=1).cpu().numpy()[0]


def predict_snr(model, iq):
    with torch.no_grad():
        x = torch.tensor(iq[np.newaxis, ...], dtype=torch.float32).to(DEVICE)
        return float(model(x).cpu().numpy()[0])


if not DATA_PATH.exists():
    st.error("data/signals.mat was not found. Run matlab/generate_signals.m first.")
    st.stop()

X, labels, snrs = load_signals()
models = load_trained_models()

with st.sidebar:
    st.header("Signal Selection")
    target_mod = st.selectbox("True modulation", sorted(set(labels)))
    target_snr = st.slider("SNR dB", int(np.min(snrs)), int(np.max(snrs)), 10, step=2)

candidates = np.where((labels == target_mod) & (snrs == target_snr))[0]
if len(candidates) == 0:
    st.warning("No signal found for that combination.")
    st.stop()

with st.sidebar:
    sample_idx = st.slider("Sample #", 1, len(candidates), 1) - 1
    threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.60, step=0.05)
    model_name = st.selectbox("Classifier", ["CLDNN Lite", "CNN Baseline"])
    representation = st.radio("View", ["IQ", "Amplitude-Phase", "Spectrogram", "Constellation"], horizontal=False)

idx = int(candidates[sample_idx])
iq = X[idx]

col1, col2, col3 = st.columns(3)
col1.metric("True modulation", target_mod)
col2.metric("True SNR", f"{target_snr} dB")
col3.metric("Sample index", idx)

if models is None:
    st.info("Models are not trained yet. Run: python src/train.py")
else:
    classes = np.asarray(models["metadata"]["classes"])
    probs = predict_probabilities(models[model_name], iq)
    pred, confidence = confidence_label(probs, classes, threshold)
    snr_estimate = predict_snr(models["SNR Regressor"], iq)

    p1, p2, p3 = st.columns(3)
    p1.metric("Prediction", pred)
    p2.metric("Confidence", f"{confidence * 100:.1f}%")
    p3.metric("Estimated SNR", f"{snr_estimate:.1f} dB")

    LOG_PATH.parent.mkdir(exist_ok=True)
    log_row = pd.DataFrame([
        {
            "true_label": target_mod,
            "predicted_label": pred,
            "true_snr_db": target_snr,
            "estimated_snr_db": round(snr_estimate, 2),
            "confidence": round(confidence, 4),
            "model": model_name,
        }
    ])
    existing = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
    pd.concat([existing, log_row], ignore_index=True).tail(200).to_csv(LOG_PATH, index=False)

st.subheader("Signal Views")
if representation == "Constellation":
    fig, ax = plt.subplots(figsize=(5, 5))
else:
    fig, ax = plt.subplots(figsize=(10, 3))

if representation == "IQ":
    ax.plot(iq[:, 0], label="I")
    ax.plot(iq[:, 1], label="Q")
    ax.set_title("IQ waveform")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
elif representation == "Amplitude-Phase":
    z = iq[:, 0] + 1j * iq[:, 1]
    ax.plot(np.abs(z), label="Amplitude")
    ax.plot(np.unwrap(np.angle(z)), label="Phase")
    ax.set_title("Amplitude and phase")
    ax.set_xlabel("Sample")
    ax.legend()
elif representation == "Constellation":
    ax.scatter(iq[:, 0], iq[:, 1], alpha=0.7, c='blue', edgecolors='none', s=15)
    ax.set_title("Constellation Diagram (I vs Q)")
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
else:
    f, t, power = make_spectrogram(iq)
    extent = [t[0], t[-1], f[0], f[-1]]
    img = ax.imshow(
        power, aspect="auto", origin="lower", extent=extent,
        interpolation="bilinear", cmap="inferno",
    )
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax, label="dB")

if representation == "Constellation":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)
else:
    st.pyplot(fig)

st.subheader("FFT Spectrum")
freqs, mag = fft_spectrum(iq)
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(freqs, mag)
ax2.set_xlabel("Frequency Hz")
ax2.set_ylabel("Normalized magnitude")
st.pyplot(fig2)

st.subheader("Model Comparison")
st.caption("This graph displays the global accuracy across all SNRs evaluated during training/testing. It summarizes the overall model performance and does not change based on individual signal selection.")
acc_files = list(MODELS_DIR.glob("*_snr_accuracy.csv"))
if acc_files:
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    for file in acc_files:
        df = pd.read_csv(file)
        ax3.plot(df["snr_db"], df["accuracy"], marker="o", label=file.stem.replace("_snr_accuracy", ""))
    ax3.set_xlabel("SNR dB")
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    st.pyplot(fig3)
else:
    st.info("Accuracy curves appear after training.")

st.subheader("Confusion Matrices")
st.caption("Shows how often the true modulation (rows) was predicted as another class (columns). Evaluated on the MATLAB synthetic test set.")
cm_files = list(MODELS_DIR.glob("*_confusion_matrix.csv"))
if cm_files:
    cm_cols = st.columns(len(cm_files))
    for idx, file in enumerate(cm_files):
        with cm_cols[idx]:
            st.write(f"**{file.stem.replace('_confusion_matrix', '').upper()}**")
            df = pd.read_csv(file, index_col=0)
            df_norm = df.div(df.sum(axis=1), axis=0)
            styled = df_norm.style.background_gradient(cmap='Blues').format("{:.1%}")
            st.dataframe(styled, use_container_width=True)
else:
    st.info("Confusion matrices will appear after running generate_confusion_matrices.py")

st.subheader("Prediction Log")
if LOG_PATH.exists():
    st.dataframe(pd.read_csv(LOG_PATH).tail(20), use_container_width=True)
else:
    st.info("Prediction log will appear after the first model prediction.")
