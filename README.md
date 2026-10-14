# Early Detection of Concept Drift and Model Degradation Using Deep Sequence Models

**Research question:** Can internal prediction signals (confidence, entropy) from a deployed MLP be used by sequence models (LSTM/GRU) to *forecast* model degradation before conventional error-based drift detectors (ADWIN, DDM) react?

---

## Project Overview

This project implements a complete streaming-ML pipeline on the **Elec2 electricity market dataset** (45,312 samples). A NumPy MLP classifier runs prequentially over the stream, emitting per-timestep confidence and entropy signals. Custom NumPy LSTM and GRU models are trained to predict *future* model degradation from windows of those signals, and their performance is compared against ADWIN and DDM drift detectors from the `river` library.

All models — MLP, LSTM, GRU — are implemented from scratch in NumPy with manual backpropagation (no PyTorch/TensorFlow). Gradient correctness is verified numerically.

---

## Repository Structure

```
DLProject/
├── data/
│   └── electricity.csv          # Elec2 dataset (not committed — see below)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Load, scale, chronological 60/20/20 split
│   ├── mlp.py                   # NumPy MLP + prequential runner
│   ├── rnn_layers.py            # NumPy LSTM, GRU cells + SequenceHead
│   ├── sequence_dataset.py      # Sliding-window builder + degradation labels
│   ├── drift_detectors.py       # ADWIN / DDM wrappers + synthetic benchmarks
│   ├── evaluate.py              # Metrics, threshold tuning, all plots
│   ├── gradcheck.py             # Numerical gradient verification
│   └── train.py                 # End-to-end orchestration (10-step pipeline)
├── tests/
│   └── test_pipeline.py         # 31 unit tests (all passing)
├── reports/
│   └── figures/                 # Generated plots (after running pipeline)
├── results/                     # JSON metrics tables (after running pipeline)
├── artifacts/                   # Saved model params + monitoring arrays
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/Viz248/DLProject.git
cd DLProject
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `river`

### 3. Place the dataset

Download the Elec2 dataset and place it at `data/electricity.csv`.

The file must have these columns:
`date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer, class`

Dataset source: https://datahub.io/core/openml-datasets/data/electricity  
(45,312 rows, `class` column: UP / DOWN)

---

## Running the Pipeline

### Gradient check (verify LSTM/GRU backprop)

```bash
python -m src.gradcheck
```

Expected output:
```
[LSTM] max relative error: 0.000xxx  -> PASSED
[GRU]  max relative error: 0.000xxx  -> PASSED
```

### Full training + evaluation pipeline

```bash
python -m src.train
```

This runs all 10 steps (≈ 5–15 minutes depending on hardware):

1. Load and preprocess Elec2 with chronological 60/20/20 split
2. MLP warm-up training on training period
3. Prequential evaluation — collects confidence, entropy, error signals
4. ADWIN and DDM on the error stream
5. Build sliding-window sequence datasets
6. Compute class weights; tune thresholds on validation set
7. Train LSTM and GRU (3 seeds each, class-weighted loss)
8. Monitoring-signal ablation study
9. Synthetic drift benchmark (ADWIN vs DDM, known change points)
10. Save all artifacts, plots, and result tables

**Outputs written to:**
| Path | Contents |
|------|----------|
| `results/table_a_mlp_metrics.json` | MLP accuracy, F1, ROC-AUC, PR-AUC |
| `results/table_b_forecasting.json` | LSTM/GRU/baselines forecasting metrics |
| `results/table_c_drift_detection.json` | ADWIN/DDM detection metrics |
| `results/ablation_results.json` | Ablation by monitoring-signal subset |
| `results/experiment_metadata.json` | Full experiment config + split indices |
| `reports/figures/` | All plots (see below) |
| `artifacts/` | Model params (`.npz`), signal arrays (`.npy`) |

### Generated plots

| File | Description |
|------|-------------|
| `mlp_signals.png` | Rolling error, confidence, entropy over stream |
| `signals_detectors.png` | Signals + ADWIN/DDM alarms + LSTM/GRU probs |
| `degradation_probs.png` | LSTM & GRU P(future degradation) on test set |
| `cm_lstm.png` / `cm_gru.png` | Confusion matrices |
| `pr_lstm.png` / `pr_gru.png` | Precision-recall curves |
| `model_comparison.png` | Bar chart: all models vs baselines |
| `ablation.png` | Ablation: conf-only vs ent-only vs conf+ent |
| `synthetic_drift.png` | Known change point vs ADWIN/DDM alarms |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

**31 tests, all passing.** Covers:
- Preprocessing: shapes, no-leakage scaler, label validity, split ordering
- MLP: forward pass, probability validity, training convergence, stream runner
- Sequence dataset: shapes, no future leakage in inputs, label logic, split no-overlap
- RNN layers: LSTM/GRU forward, prediction range, loss finiteness, gradient checks, param serialisation
- Drift detectors: detection on injected drift, false-alarm rate on stationary stream, metric calculations
- Evaluate: threshold selection, metric edge cases, majority baseline

---

## Experimental Methodology

### Data splits (no leakage)
- **Train** (0–60%): MLP warm-up; scaler fitted here only
- **Gap** (40 samples): prevents sequence windows from straddling the boundary
- **Validation** (60–80%): threshold tuning and model selection
- **Gap** (40 samples)
- **Test** (80–100%): final evaluation, never touched during tuning

### Degradation label definition

For a window ending at stream position *t*:

```
past_err(t)   = mean(errors[t - window_len : t])
future_err(t) = mean(errors[t : t + horizon])
label(t)      = 1  if  future_err(t) - past_err(t) > degradation_delta
                0  otherwise
```

Defaults: `window_len=20`, `horizon=20`, `degradation_delta=0.15`

The model input is `[C_s, H_s]` for `s ∈ [t-window_len, t)` — **strictly past information**. Future error rates appear only in the label, never in model inputs.

### Drift detection vs degradation forecasting
These are **distinct tasks**:
- ADWIN/DDM detect distributional changes in the **error stream** (reactive)
- LSTM/GRU forecast **future degradation** from internal signals (proactive)

Metrics are kept separate. Detection delay (ADWIN/DDM) and positive-class F1 (LSTM/GRU) measure different things and are not directly compared numerically.

---

## Key Implementation Notes

- All three models (MLP, LSTM, GRU) are pure NumPy — no deep learning frameworks
- LSTM uses bias initialisation of 1.0 for the forget gate (standard practice to help gradient flow early in training)
- Class imbalance (~21% positive windows) is handled via `pos_weight = neg_rate / pos_rate` in the loss
- Classification threshold is tuned on validation F1, not hardcoded at 0.5
- Results on the real Elec2 stream are reported honestly — no fabricated ground-truth drift labels

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
river
pytest          # for tests only
```

Python ≥ 3.9 required. Tested on Python 3.9.13.
