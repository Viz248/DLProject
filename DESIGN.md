# System Design Document

**Project:** Early Detection of Concept Drift and Model Degradation Using Deep Sequence Models  
**Dataset:** Elec2 (Electricity Market) — 45,312 samples, binary classification (UP / DOWN)

---

## 1. Problem Statement

A deployed MLP classifier processes an electricity market data stream in real time. Over time, the underlying data distribution shifts (concept drift), causing the model's accuracy to degrade. The goal is to **forecast this degradation before it fully manifests** by learning temporal patterns in the model's own internal signals — confidence and prediction entropy — using LSTM and GRU sequence models.

This is distinct from reactive drift detection (ADWIN, DDM), which only alarm after degradation has already occurred in the error stream.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Elec2 Data Stream                        │
│              45,312 samples, chronological order                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Preprocessing Layer                        │
│  • StandardScaler fit on training period only (no leakage)      │
│  • 60% Train │ gap │ 20% Validation │ gap │ 20% Test            │
│  • Temporal gaps of 40 samples at each boundary                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLP Base Classifier                          │
│  • Architecture: Dense(64, ReLU) → Dropout(0.2) → Dense(2)     │
│  • Warm-up: batch training on training period (30 epochs)       │
│  • Stream phase: prequential test-then-train (online SGD)       │
│  • Outputs per timestep t:                                      │
│      C_t  = max(P(y=k|x_t))          [confidence]              │
│      H_t  = -Σ P log P               [entropy]                  │
│      E_t  = 1[ŷ_t ≠ y_t]            [error indicator]          │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐
    │    ADWIN    │  │     DDM     │  │  Sequence Windows  │
    │  (reactive) │  │  (reactive) │  │  window=20, hop=1  │
    │  river lib  │  │  river lib  │  │  input:[C_t, H_t]  │
    └─────────────┘  └─────────────┘  └────────┬──────────┘
                                               │
                                ┌──────────────┼──────────────┐
                                ▼              ▼              ▼
                        ┌──────────┐   ┌──────────┐   ┌──────────┐
                        │   LSTM   │   │   GRU    │   │Baselines │
                        │ n_hid=32 │   │ n_hid=32 │   │ Majority │
                        │ 3 seeds  │   │ 3 seeds  │   │ RollEnt  │
                        └──────────┘   └──────────┘   └──────────┘
                                │              │
                                └──────┬───────┘
                                       ▼
                        ┌──────────────────────────┐
                        │     Evaluation Layer     │
                        │  Table A: MLP metrics    │
                        │  Table B: Forecasting    │
                        │  Table C: Drift detect.  │
                        │  Ablation study          │
                        └──────────────────────────┘
```

---

## 3. Data Pipeline

### 3.1 Chronological Split

| Split | Range | Samples | Purpose |
|-------|-------|---------|---------|
| Train | 0 – 60% | ~27,187 | MLP warm-up, scaler fit, RNN training |
| Gap | 40 samples | 40 | Prevents window boundary leakage |
| Validation | 60–80% | ~9,062 | Threshold tuning, model selection |
| Gap | 40 samples | 40 | Prevents window boundary leakage |
| Test | 80–100% | ~9,023 | Final evaluation only |

The temporal gaps ensure no sliding window of length 20 can straddle a split boundary and pull in data from a different period.

### 3.2 Feature Scaling

`StreamScaler` (z-score normalisation) is fitted **exclusively** on the training period and then applied to the full stream. This mirrors production deployment — a scaler can only be fit on data seen before launch.

### 3.3 Degradation Label Definition

For each window ending at stream position *t*:

```
past_err(t)   = mean( E[t - window_len : t] )
future_err(t) = mean( E[t : t + horizon] )

label(t) = 1   if  future_err(t) − past_err(t) > δ
           0   otherwise
```

**Parameters:** `window_len = 20`, `horizon = 20`, `δ = 0.15`

The model **input** is `[C_s, H_s]` for `s ∈ [t−window_len, t)` — only past signals.  
The **label** uses future error rates, which are never placed in the input (no leakage).

---

## 4. Model Architectures

### 4.1 MLP Base Classifier

```
Input(8 features)
    → Dense(64) → ReLU → Dropout(0.2)
    → Dense(2)  → Softmax
```

- **Training:** batch SGD (lr=0.003, batch=64, epochs=30) on training period
- **Stream phase:** prequential test-then-train with online SGD

### 4.2 LSTM Degradation Forecaster

```
Input: (window_len=20, n_features=2)  ← [confidence, entropy] sequence
    → LSTMCell(n_hidden=32)           ← custom NumPy implementation
    → h_T (final hidden state)
    → Linear(32 → 1) → Sigmoid
Output: P(future degradation) ∈ [0, 1]
```

**LSTM gate equations:**
```
z_t  = [h_{t-1} ; x_t]
f_t  = σ(z_t · Wf + bf)        forget gate
i_t  = σ(z_t · Wi + bi)        input gate
g_t  = tanh(z_t · Wc + bc)     candidate
o_t  = σ(z_t · Wo + bo)        output gate
c_t  = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t  = o_t ⊙ tanh(c_t)
```

Forget gate bias initialised to 1.0 (standard practice to encourage gradient flow early in training).

### 4.3 GRU Degradation Forecaster

```
Input: (window_len=20, n_features=2)
    → GRUCell(n_hidden=32)
    → h_T
    → Linear(32 → 1) → Sigmoid
Output: P(future degradation) ∈ [0, 1]
```

**GRU gate equations:**
```
z_in    = [h_{t-1} ; x_t]
r_t     = σ(z_in · Wr + br)           reset gate
u_t     = σ(z_in · Wu + bu)           update gate
z_cand  = [r_t ⊙ h_{t-1} ; x_t]
h̃_t    = tanh(z_cand · Wh + bh)      candidate
h_t     = (1 − u_t) ⊙ h_{t-1} + u_t ⊙ h̃_t
```

### 4.4 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden units | 32 | Sufficient capacity; larger showed overfitting on validation |
| Learning rate | 0.005 | Stable convergence across seeds |
| Epochs | 10 | Validated against val loss curve; no improvement after epoch 8 |
| Gradient clip | 1.0 | Prevents BPTT explosion over 20 steps |
| Class weight | neg_rate / pos_rate | ~79% negative vs ~21% positive |
| Threshold | Val-F1 sweep | Not hardcoded at 0.5 |
| Seeds | 0, 1, 2 | Reports mean ± std |

---

## 5. Baseline Models

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Majority class | Always predicts class 0 (no degradation) | Lower bound |
| Rolling entropy | Rolling mean of H_t ≥ threshold | Simple signal baseline |
| ADWIN | Adaptive windowing on error stream | Reactive drift detector |
| DDM | Drift Detection Method on error stream | Reactive drift detector |

All baselines use the same evaluation protocol on the same test windows as LSTM/GRU.  
Rolling entropy threshold is tuned on validation data only.

---

## 6. Evaluation Protocol

### Classification Metrics (Table B — Forecasting)
- Positive-class precision, recall, F1
- PR-AUC (preferred over ROC-AUC for imbalanced data)
- Confusion matrix
- Threshold used
- Positive-label prevalence

### Drift Detection Metrics (Table C)
- Number of alarms
- True detections, missed detections
- Mean detection delay (samples after known change point)
- False-alarm rate
- Runtime (seconds)

These two metric sets are **not directly compared** — they measure fundamentally different things.

---

## 7. Ablation Study Design

Four LSTM variants trained and evaluated identically:

| Variant | Input features |
|---------|---------------|
| Conf only | Confidence C_t |
| Ent only | Entropy H_t |
| Conf + Ent | Both (baseline configuration) |
| Full (if extra features) | All available features |

Same seeds, same threshold tuning, same evaluation split across all variants.

---

## 8. Output Artifacts

| Location | Contents |
|----------|----------|
| `results/table_a_mlp_metrics.json` | MLP accuracy, F1, ROC-AUC, PR-AUC |
| `results/table_b_forecasting.json` | All forecasting model metrics |
| `results/table_c_drift_detection.json` | ADWIN/DDM detection metrics |
| `results/ablation_results.json` | Ablation results per signal subset |
| `results/experiment_metadata.json` | Full config, split indices, hyperparams |
| `artifacts/lstm_params.npz` | Best LSTM weights |
| `artifacts/gru_params.npz` | Best GRU weights |
| `artifacts/confs.npy` | Confidence signal array |
| `artifacts/ents.npy` | Entropy signal array |
| `artifacts/errors.npy` | Error indicator array |
| `reports/figures/*.png` | All visualisation plots |
