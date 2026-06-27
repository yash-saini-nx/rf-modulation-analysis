#  RF Modulation Analysis & Sim-to-Real Gap

This project is an advanced, cross-disciplinary Machine Learning and Digital Signal Processing (DSP) system. It teaches a computer to recognize the modulation type of an RF (Radio Frequency) signal by analyzing its raw, complex-valued Time-Series I/Q data. 

Unlike basic ML projects, this system addresses the **Sim-to-Real Gap**—training models on clean synthetic mathematical signals generated in MATLAB, and evaluating their performance drop when exposed to real-world benchmark RF signals (RadioML 2016.10A).

---

##  The Big Idea

Modulation is how information is placed onto a carrier wave (e.g., AM changes amplitude, FM changes frequency, QAM changes both). Modern communication systems represent these RF signals internally as **IQ samples** (In-phase and Quadrature), which behave like x and y coordinates of a rotating signal point.

This project uses **MATLAB** as a "signal factory" to generate clean mathematical RF signals and add AWGN (Additive White Gaussian Noise). It then uses **Python (PyTorch)** to build Deep Learning models that classify these signals, and **Streamlit** to provide a rich, interactive visual dashboard.

---

##  Key Features (Beyond a Basic ML Project)

- **Multiple Deep Learning Architectures:** Compares a standard CNN baseline with a CLDNN-lite (CNN + LSTM) model to capture sequence dependencies in RF data.
- **Sim-to-Real Gap Analysis:** Includes a Jupyter notebook that takes the model trained on synthetic MATLAB data and tests it against the real-world **RadioML 2016.10A** dataset, successfully demonstrating the accuracy drop caused by real-world channel impairments.
- **SNR Regression:** Alongside classification, an independent regression model predicts the Signal-to-Noise Ratio (SNR) of the incoming signal in real-time.
- **Confidence Rejection:** Implements a dynamic confidence threshold. Instead of making bad guesses on highly noisy signals, the model will output `"unclassifiable"`.
- **Interactive Dashboard:** A professional Streamlit UI that visualizes Time-Domain I/Q waveforms, Amplitude-Phase plots, FFT Spectrums, Spectrograms, and **Constellation Diagrams** alongside real-time model predictions and Pandas logging.
- **Confusion Matrices:** Evaluates models on the test set and visualizes confusion matrices as color-coded heatmaps to analyze exactly which modulations the models struggle to differentiate.

---

##  Project Architecture

### Signal Generation (MATLAB)
- **`matlab/generate_signals.m`**: Generates 7 modulation types (AM, FM, SSB, BPSK, QPSK, QAM16, and QAM64). It adds AWGN noise at multiple SNR values (-10 dB to +20 dB) and saves the I/Q data into `data/signals.mat`.

### Deep Learning Pipeline (Python)
- **`src/dataset.py`**: Loads the MATLAB `.mat` file, converts string labels to categorical indices, normalizes I/Q values, and builds PyTorch DataLoaders.
- **`src/preprocess.py`**: Contains DSP helper functions. Converts I/Q to complex signals, calculates FFT spectrums, generates Spectrograms, and applies confidence rejection.
- **`src/model.py`**: Defines three PyTorch models: a CNN baseline, a CLDNN-lite classifier, and an SNR regressor. 
- **`src/train.py`**: Trains all three models, saving the weights and logging accuracy-vs-SNR results for comparison.

### Visualization & Validation
- **`app.py`**: The Streamlit dashboard. Allows you to select target modulations, visualize the signals in multiple DSP domains, and see the model's live predictions and logs.
- **`notebooks/radioml_validation.ipynb`**: Loads the external RadioML dataset, maps the labels, evaluates the synthetic-trained models, and plots the Sim-to-Real Gap.

---

##  How To Run

### 1. Generate Signals In MATLAB
Open MATLAB, navigate to the `matlab` folder, and run:
```matlab
generate_signals
```
*This creates the dataset at `data/signals.mat`.*

### 2. Install Python Libraries
From the root project folder, install the dependencies:
```bash
pip install -r requirements.txt
```
*(If PyTorch installation fails on your system, install PyTorch directly from the [official PyTorch install page](https://pytorch.org/get-started/locally/) for your specific CUDA/CPU setup).*

### 3. Train the Models
Train the CNN, CLDNN, and SNR Regressor:
```bash
python src/train.py
```
*This will populate the `models/` folder with `.pt` weight files, accuracy CSVs, and metadata.*

### 4. Start the Dashboard
Launch the interactive web UI:
```bash
streamlit run app.py
```

### 5. Evaluate the Sim-to-Real Gap (Optional)
To see the model fail on real data:
1. Download `RML2016.10a_dict.pkl` from Kaggle or DeepSig.
2. Place it in the `data/` folder.
3. Open and run all cells in `notebooks/radioml_validation.ipynb`.

---

##  Future Extensions

1. **Transfer Learning:** Fine-tune the MATLAB-trained model on a small portion of the RadioML dataset to close the sim-to-real gap.
2. **Advanced Augmentation:** Add realistic channel impairments to the MATLAB script (Multipath Fading, Clock Drift, Phase Noise) to achieve Domain Randomization.
3. **RF-Transformers:** Replace the CLDNN with a modern Transformer architecture (Multi-Head Attention) for sequence modeling.
4. **Live SDR Integration:** Connect an RTL-SDR or HackRF to stream live over-the-air (OTA) radio waves directly into the Streamlit dashboard for real-time classification.

---
## License

This project is open-source under the [MIT License](LICENSE).