# Early Detection of Concept Drift and Model Degradation Using Deep Sequence Models

**Research question:** can internal prediction signals (confidence, entropy) from a deployed MLP be used by sequence models (LSTM/GRU) to *forecast* model degradation before conventional error-based drift detectors (ADWIN, DDM) react?

A NumPy MLP classifier runs prequentially over the **Elec2 electricity dataset** (45,312 samples). Custom NumPy LSTM and GRU models (manual backpropagation, no PyTorch/TensorFlow, gradient-checked) read windows of the MLP's confidence/entropy and predict whether its error rate is about to rise. They are compared with simple baselines and with ADWIN/DDM from `river`.

---

## Repository structure

```
data/electricity.csv        Elec2 dataset
src/preprocessing.py        loading, causal scaling, warm-up / stream split
src/mlp.py                  NumPy MLP + prequential runner
src/rnn_layers.py           NumPy LSTM, GRU cells + readout head
src/sequence_dataset.py     sliding windows, degradation labels, window split
src/drift_detectors.py      ADWIN / DDM wrappers + synthetic benchmarks
src/early_warning.py        event-level early-warning evaluation (new)
src/label_experiment.py     validation-only comparison of label reference windows
src/demo.py                 drift demo: inject known drift, run the system, give a verdict
demo/index.html             demo dashboard (reads demo/demo_data.js)
src/evaluate.py             metrics, baselines, threshold tuning, plots
src/gradcheck.py            numerical gradient verification
src/train.py                end-to-end pipeline
src/report.py               print the saved result tables
tests/test_pipeline.py      50 unit / regression tests
docs/results/               snapshot of the results and figures from the final run
docs/results/label_experiment/  label experiment (validation table + the R=100 test run)
```

## Setup and running

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # 50 tests
python -m src.gradcheck             # LSTM/GRU gradient check
python -m src.train                 # full pipeline (~11-25 min on one CPU core)
python -m src.train --quick         # ~1 min smoke run
python -m src.train --resume        # reuse trained LSTM/GRU + cached ablation runs
python -m src.report                # print the tables from results/
python -m src.label_experiment      # label look-back comparison, validation only (~20 min)
python -m src.train --reference-len 100   # full pipeline with the longer-reference label
```

Outputs go to `results/` (JSON tables), `reports/figures/` (plots) and `artifacts/` (model parameters, signal arrays). These folders are gitignored because they are regenerable; the committed snapshot of the final run is in `docs/results/`.

---

## Demo: drift / no drift / uncertain

```bash
python -m src.demo                                    # all 9 scenarios, ~2 min (needs a finished src.train run)
python -m src.demo --scenario sudden --severity 0.25  # one scenario in the terminal
```

Then open `demo/index.html` in a browser (it reads `demo/demo_data.js`, which is committed, so the dashboard works without rerunning).

The demo replays the real system on the Elec2 test stream. At test step 4,000 it injects drift of a known type and size; the MLP keeps predicting and learning online, ADWIN and DDM watch its errors, and the trained LSTM reads its confidence/entropy. The 2,000 steps before the injection are compared with the 2,000 after:

* **Scenarios:** no drift (real stream); sudden and gradual concept drift (the up/down rule reverses during 10 / 25 / 50 % of the day's half-hour slots); feature shift (price inputs +1 / +2.5 standard deviations, labels unchanged).
* **DRIFT** - the error rate rose significantly (moving-block bootstrap 95 % CI above 0), by at least 5 percentage points, and ADWIN or DDM fired. Severity: mild < 10 pp, moderate 10-20 pp, severe > 20 pp.
* **NO DRIFT** - no significant rise, estimated change < 5 pp, no ADWIN/DDM alarm.
* **UNCERTAIN** - the evidence disagrees (e.g. a rise no detector confirms, or an alarm with a small change).
* The output reports accuracy before -> after (pp and relative), the error-rate change with its CI, the worst 200-step error rate, each detector's delay, and the label-free signals.

What the demo shows: every injected concept drift is called DRIFT, with accuracy drops that grow with the injected size, and ADWIN reacts faster than DDM in every case but the weakest gradual drift (where they tie within 10 steps). The feature shifts come out UNCERTAIN (small accuracy cost, detectors mostly quiet). The real stream comes out NO DRIFT. The LSTM's alarm rate does **not** rise under injected drift, which matches the main finding that the forecaster gives no early warning; the MLP's confidence does drop under strong concept drift.

---

## Method

**Split (no leakage).** The MLP is warm-up trained offline on the first 10 % of the data only. The prequential (test-then-train) pass then runs over the remaining 90 % and produces confidence, entropy and error signals for the whole monitored stream. Windows built from that stream are split 60 / 20 / 20 by window centre with 40-sample gaps: **24,448 train / 8,116 validation / 8,097 test windows.** Features are scaled causally: row *t* is standardised with the mean/std of rows before *t* only (clipped at ±5σ).

**Degradation label.** For a window ending at stream position *t*: `label = 1` if `mean(err[t:t+20]) - mean(err[t-20:t]) > 0.15`. The model input is `[confidence, entropy]` over `[t-20, t)`, strictly past data. Positive rate: 0.142 train, 0.190 validation, 0.125 test.

**Training.** Per-sample SGD, 8 epochs, lr 0.002 with decay, hidden size 32, half-strength class weighting, 3 seeds each. The epoch is chosen by validation PR-AUC and the decision threshold by validation F1. Hyperparameters (learning rate, class-weight strength, hidden size) were chosen on validation PR-AUC only; the test split was used once for reporting.

**Detection vs forecasting.** ADWIN/DDM react to the error stream; LSTM/GRU forecast from internal signals. Table B compares forecasting quality; Table D compares them on the same degradation events.

---

## Results (test split, single final run)

### Table A - MLP on the test stream
accuracy 0.8928 · precision 0.8834 · recall 0.8839 · F1 0.8836 · ROC-AUC 0.9534 · PR-AUC 0.9444 (prequential accuracy over the whole monitored stream: 0.8677).

### Table B - degradation forecasting (positive class; test positive rate 0.125)

| Model | Precision | Recall | F1 | PR-AUC | Accuracy |
|---|---|---|---|---|---|
| Majority (never warn) | 0.000 | 0.000 | 0.000 | 0.125 | 0.875 |
| Always warn | 0.125 | 1.000 | 0.222 | 0.125 | 0.125 |
| Low past error rate * | 0.162 | 0.881 | 0.274 | 0.177 | 0.418 |
| Rolling entropy | 0.082 | 0.228 | 0.121 | 0.093 | 0.586 |
| Rolling low confidence | 0.076 | 0.228 | 0.114 | 0.090 | 0.556 |
| Gradient boosting on window summaries (reference) | 0.245 | 0.445 | 0.316 | 0.250 | 0.760 |
| **LSTM** (mean of 3 seeds) | 0.237 | 0.494 | 0.317 | 0.238 | 0.736 |
| LSTM, high-precision point | 0.252 | 0.161 | 0.197 | 0.233 | 0.836 |
| **GRU** (mean of 3 seeds) | 0.243 | 0.468 | 0.315 | 0.227 | 0.749 |
| GRU, high-precision point | 0.244 | 0.165 | 0.197 | 0.216 | 0.832 |

LSTM across 3 seeds: F1 0.317 ± 0.013, precision 0.237 ± 0.010, PR-AUC 0.238 ± 0.004. GRU: F1 0.315 ± 0.013, precision 0.243 ± 0.013, PR-AUC 0.227 ± 0.008.
`*` needs ground-truth labels of the recent past; the other rows use only the MLP's internal signals. The high-precision rows use the validation threshold with the best precision subject to recall >= 0.25.

### Ablation (LSTM, 3 seeds)

| Inputs | F1 | PR-AUC |
|---|---|---|
| Conf only | 0.298 ± 0.002 | 0.287 ± 0.010 |
| Ent only | 0.287 ± 0.008 | 0.230 ± 0.040 |
| Conf+Ent | 0.317 ± 0.013 | 0.238 ± 0.004 |

### Table D - early warning on severe-degradation events (test stream)
Events are onsets where the next-20-step error rate exceeds the past-20-step rate by more than 0.30 (25 events, merged within 40 steps). An alarm counts as a hit if it falls in `[onset - 30, onset + 60]`. LSTM/GRU alarm when their output exceeds the 95th percentile of their own previous 2,000 outputs (causal). "Chance" is the expected recall of the same number of random alarms.

| Method | Alarms | Event recall | Chance | Warned before onset | Median offset (steps) | False alarms / 1000 |
|---|---|---|---|---|---|---|
| LSTM | 59 | 0.52 | 0.47 | 0.28 | -3 | 5.1 |
| GRU | 58 | 0.52 | 0.47 | 0.20 | 2 | 5.2 |
| ADWIN | 1 | 0.04 | 0.01 | 0.04 | -28 | 0.0 |
| ADWIN(d=.05) | 10 | 0.24 | 0.10 | 0.04 | 38 | 0.5 |
| DDM | 0 | 0.00 | 0.00 | 0.00 | n/a | 0.0 |

### Synthetic benchmark (known change point)
ADWIN detected every change point with a delay of 47-111 steps and no false alarms; DDM detected them with a delay of 112-270 steps and raised one false alarm on a stationary stream (`results/table_c_drift_detection.json`).

---

## Honest reading of the results

* **The first version of this project was invalid.** It left about 20 training windows for the forecasters (see the fix log below). With a proper training set, test F1 rose from 0.24 to 0.32, precision from 0.14 to 0.24 and accuracy from 0.42 to 0.74.
* **The forecasters have real but modest skill.** PR-AUC is about 0.24 against a 0.125 base rate, and F1 is 0.32. LSTM and GRU are statistically indistinguishable, and a gradient-boosting model on window summaries performs about the same, so the benefit of the recurrent models is not demonstrated here.
* **Precision can be traded for recall.** The high-precision operating point reaches precision 0.25 and accuracy 0.84, but recall drops to 0.16.
* **Early warning is not demonstrated.** On severe-degradation events the forecasters hit 0.52 of events versus 0.47 for random alarms - within noise for 25 events. ADWIN/DDM rarely fire on the test stream, and when ADWIN (delta 0.05) does, it is late (median +38 steps).
* **Label caveat.** Because the label is "future minus past" error, it partly measures mean reversion: a baseline that warns when the *past* error is low already reaches PR-AUC 0.177. A longer reference window or an absolute error-rate label would be a sensible next experiment.
* Positive-rate shift (0.14 / 0.19 / 0.125) means thresholds tuned on validation transfer imperfectly to test.

---

## Label experiment: longer reference window

The label above compares the next 20 steps with the previous 20, so it partly
rewards mean reversion. `make_windows(reference_len=R)` instead compares the
next 20 steps with `mean(err[t-R:t])`; inputs and windows are unchanged.

**Selection rule (fixed before running, validation only).** Score each R by
`(AP_LSTM - AP_LowPastErr*) / positive rate` on the odd validation windows
(the even ones are used for early stopping), 2 LSTM seeds. Keep R = 20 unless
the best R beats it by more than the seed spread.

| R | Val pos. rate | LSTM AP | LowPastErr* AP | GBM AP | Skill |
|---|---|---|---|---|---|
| 20 (original) | 0.188 | 0.401 ± 0.016 | 0.337 | 0.406 | 0.342 ± 0.083 |
| **100** | 0.170 | 0.312 ± 0.007 | 0.235 | 0.329 | **0.454 ± 0.043** |
| 500 | 0.161 | 0.260 ± 0.005 | 0.218 | 0.260 | 0.259 ± 0.032 |
| 2000 | 0.165 | 0.218 ± 0.009 | 0.202 | 0.255 | 0.096 ± 0.057 |

R = 100 was selected (gain +0.111 against a seed spread of 0.083, so only just).
The full pipeline was then run **once** on test with R = 100
(`docs/results/label_experiment/`):

| Test split | R = 20 (main) | R = 100 |
|---|---|---|
| Positive rate | 0.125 | 0.124 |
| LowPastErr* PR-AUC | 0.177 | 0.176 |
| GBM-summary PR-AUC / F1 | 0.250 / 0.316 | 0.207 / 0.281 |
| LSTM PR-AUC / F1 / precision | 0.238 / 0.317 / 0.237 | 0.233 / 0.303 / 0.221 |
| GRU PR-AUC / F1 / precision | 0.227 / 0.315 / 0.243 | 0.193 / 0.269 / 0.212 |
| LSTM skill `(AP - AP_LowPastErr*)/rate` | 0.49 | 0.46 |
| Early warning, LSTM recall vs chance | 0.52 vs 0.47 (25 events) | 0.57 vs 0.45 (23 events) |
| Early warning, GRU recall vs chance | 0.52 vs 0.47 | 0.52 vs 0.55 |

**Result: the longer reference did not help on test.** On validation it weakened
the mean-reversion baseline (LowPastErr* lift 1.79 -> 1.38), but on test that
baseline is unchanged (PR-AUC 0.177 vs 0.176) and no model improved; GRU and
GBM got worse. The validation gain did not transfer, which is consistent with
it being only just above the seed noise. The main results above therefore stay
on the original label (`reference_len=None`); the R = 100 run is reported as a
negative result. Early warning is still not demonstrated with either label.

---

## Fix log (what changed from the first version)

1. **Training-set bug.** The MLP was warmed up on 60 % of the stream, so monitoring signals existed only for the last 40 %, and the "train" window split contained about 20 windows. Now: 10 % warm-up, causal scaling, and the windows split 60/20/20 of the monitored stream.
2. **Leaky baseline removed.** The old rolling-entropy baseline tuned its threshold on the test labels and used a centred window that sees future values. Replaced by validation-tuned baselines.
3. **New baselines** (always-warn, past error, rolling entropy/confidence, gradient boosting) and PR-AUC for the trivial baselines.
4. **Early-warning evaluation** against ADWIN/DDM with a random-alarm chance reference (`src/early_warning.py`).
5. **Window-split boundaries** are explicit, so the label region of one split never overlaps the next split's inputs.
6. Early stopping on validation PR-AUC, input standardisation, class-weight strength and learning rate chosen on validation.
7. Removed the broken `src/report.py` (imported functions that did not exist) and rewrote it; deleted `temp_train_output.txt`.
8. Tests: 31 -> 44, including regression tests for the training-set size, causal scaling and split independence.
9. Optional `reference_len` for the degradation label, `--reference-len` CLI flag, validation-only label experiment (`src/label_experiment.py`), 2 more tests (46).

Requirements: Python >= 3.9; `numpy pandas scikit-learn matplotlib river pytest`.
