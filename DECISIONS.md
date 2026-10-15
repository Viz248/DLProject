# Design Decisions Log

**Project:** Early Detection of Concept Drift and Model Degradation Using Deep Sequence Models

This document records every significant design choice, the alternatives considered, and the justification for the decision taken. All decisions are traceable to code in `src/`.

---

## D1 — Pure NumPy implementation (no PyTorch / TensorFlow)

**Decision:** Implement MLP, LSTM, and GRU entirely in NumPy with manual backpropagation.

**Alternatives considered:**
- PyTorch with `nn.LSTM` and `nn.GRU`
- TensorFlow/Keras

**Justification:**
- No GPU or compatible deep-learning framework was available in the deployment environment
- Manual implementation allows exact inspection of every gradient calculation during the viva
- Gradient correctness is verified numerically in `src/gradcheck.py` — something harder to assert with framework black boxes
- The dataset (45K samples, 2 input features per step) is small enough that NumPy is fast enough

**Trade-off:** Slower to iterate on architecture changes; no automatic differentiation safety net.

---

## D2 — Chronological 60/20/20 split with temporal gaps

**Decision:** Split the stream chronologically into 60% train, 20% validation, 20% test, with a 40-sample gap at each boundary.

**Alternatives considered:**
- Single 80/20 train-test split (original code)
- Random cross-validation folds
- Walk-forward (expanding window) validation

**Justification:**
- The stream is temporal — random splits would create massive data leakage (future samples used to train on past evaluation)
- A validation set is mandatory for honest threshold tuning; without it, the threshold effectively sees test labels
- Walk-forward validation is the gold standard but requires many pipeline runs; 3-seed evaluation is a practical compromise
- The 40-sample gap equals twice the window length (20), ensuring no single sliding window can span both sides of a split boundary

**Trade-off:** Reduces training data to 60%. With 45K samples this is acceptable (~27K train samples).

---

## D3 — Scaler fit on training period only

**Decision:** `StreamScaler` (z-score) is fit on `X[:train_end]` only, then applied to the full stream.

**Alternatives considered:**
- Fit on the full dataset (original code used `warmup_frac=0.1` but transformed all data)
- Per-batch online normalisation

**Justification:**
- Fitting on the full dataset leaks future statistics into the training and validation transforms — a common but subtle error
- In production, a scaler can only be calibrated on data available before deployment
- Z-score is preferred over min-max for this dataset because `nswprice` and `vicprice` have extreme outliers that would compress the bulk of the distribution with min-max scaling

**Trade-off:** Validation and test features are scaled with training statistics, so their actual distributions may differ slightly. This is correct behaviour — it reflects production reality.

---

## D4 — Degradation label: future vs past rolling error rate

**Decision:** `label(t) = 1 if future_err(t) − past_err(t) > 0.15`, with `window_len=20`, `horizon=20`.

**Alternatives considered:**
- Absolute future error rate threshold (e.g. `future_err > 0.4`)
- KL divergence between past and future error distributions
- Cumulative sum (CUSUM) of errors

**Justification:**
- A relative threshold (delta) is more robust to different base error rates across stream regions
- `window_len=20` and `horizon=20` are the original project settings; changing them without experimental justification would be arbitrary
- The delta of 0.15 was retained from the original as the baseline; ablation over delta values is a future extension
- The absolute threshold alternative doesn't capture *worsening* — a model with 40% error rate is bad but stable; we care about transitions

**Key invariant enforced in code:** The input window `X[t-window_len:t]` contains only indices strictly less than `t`. Future error rates are used only to compute the label, never placed in the input tensor.

---

## D5 — Class-weighted loss for LSTM/GRU

**Decision:** Multiply the gradient by `pos_weight = neg_rate / pos_rate` for positive-class training examples.

**Alternatives considered:**
- Oversampling positive windows (SMOTE or random duplication)
- Undersampling negative windows
- No adjustment (original code)

**Justification:**
- The positive-label rate is approximately 21%, creating a 4:1 imbalance
- The original code predicted the majority class almost exclusively (near-zero positive recall)
- Class-weighted loss is the least invasive fix: no data augmentation, no sampling, consistent with how frameworks like PyTorch implement `pos_weight` in `BCEWithLogitsLoss`
- Oversampling in a temporal dataset risks duplicating windows that include correlated data, which inflates performance estimates

**Trade-off:** May increase false positives. The validation-set threshold sweep compensates by finding the operating point that maximises F1 rather than accuracy.

---

## D6 — Threshold tuning on validation set, not test set

**Decision:** Sweep thresholds from 0.05 to 0.95 on validation probabilities to maximise positive-class F1, then apply the best threshold to the test set.

**Alternatives considered:**
- Fixed threshold of 0.5 (original code)
- Youden's J statistic (sensitivity + specificity − 1)
- Precision-recall breakeven point

**Justification:**
- With 21% positive prevalence, 0.5 is not optimal — it under-predicts the minority class
- F1 balances precision and recall and is appropriate for imbalanced binary classification
- The threshold **must** be selected on validation data; selecting it on test data is a form of leakage that inflates reported metrics
- Youden's J is ROC-based and less interpretable for imbalanced problems than F1

---

## D7 — Multiple random seeds (3 seeds)

**Decision:** Train each model (LSTM, GRU, ablation variants) with seeds 0, 1, 2. Report mean ± standard deviation.

**Alternatives considered:**
- Single seed (original code)
- 5 or 10 seeds

**Justification:**
- A single seed does not reveal training instability; reported numbers may be lucky or unlucky
- 3 seeds is the minimum reasonable for mean ± std reporting in an academic context
- 5+ seeds would triple the training time with diminishing return on a dataset of this size
- All seeds use the same train/val/test split — only weight initialisation and minibatch ordering differ

---

## D8 — LSTM forget-gate bias initialised to 1.0

**Decision:** Set `self.bf = np.ones(n_hidden) * 1.0` instead of zeros.

**Alternatives considered:**
- Zero initialisation (original code)
- Orthogonal weight initialisation

**Justification:**
- Forget-gate bias of 1.0 is recommended by Jozefowicz et al. (2015) "An Empirical Exploration of Recurrent Network Architectures"
- With zero bias, the forget gate starts at σ(0)=0.5, meaning the cell discards half its memory from the first step
- With bias=1.0, the gate starts at σ(1)≈0.73, preserving more initial memory and helping gradient flow during early training
- This is a free improvement with no hyperparameter cost

---

## D9 — Gradient clipping at 1.0 for BPTT

**Decision:** Clip all parameter gradients to [−1, 1] during LSTM/GRU backpropagation through time.

**Alternatives considered:**
- Clip at 5.0 (MLP uses this)
- Global norm clipping
- No clipping

**Justification:**
- BPTT over 20 steps can cause exponential gradient growth (vanishing/exploding gradients)
- A clip threshold of 1.0 is conservative and appropriate for the small learning rate (0.005)
- The MLP clips at 5.0 because it processes single samples with no temporal unrolling
- Numerical gradient checks (`src/gradcheck.py`) confirm the analytic gradients are correct before clipping modifies them

---

## D10 — Separate evaluation for drift detection vs degradation forecasting

**Decision:** Keep Table B (LSTM/GRU metrics) and Table C (ADWIN/DDM metrics) entirely separate. Do not compare F1 scores to detection delays.

**Alternatives considered:**
- Unified "lead time" metric comparing all methods on the same timeline
- Treating LSTM alarm (prob ≥ threshold) as equivalent to ADWIN alarm

**Justification:**
- ADWIN/DDM are *detectors*: they alarm when a distributional change has already occurred in the error stream
- LSTM/GRU are *forecasters*: they predict whether degradation will occur in the next 20 steps
- These answer different questions and have incompatible output semantics
- A detection delay measured from a known synthetic change point is a valid metric for detectors
- Positive-class F1 on the degradation-label task is the correct metric for forecasters
- Mixing them (e.g. "LSTM detected drift 15 steps earlier than ADWIN") is only valid when there is a known ground-truth change point with a rigorous matching rule

**For real Elec2 stream:** No fabricated ground-truth drift indices. ADWIN/DDM alarm counts are reported as observations, not as validated detections.

---

## D11 — Rolling-entropy baseline uses past observations only

**Decision:** `np.convolve(signal, np.ones(window)/window, mode="same")` with threshold tuned on validation.

**Alternatives considered:**
- Future-aware smoothing (e.g. centred moving average)
- EWMA (exponential weighted moving average)

**Justification:**
- `mode="same"` in convolution uses symmetric padding at boundaries, which technically introduces a small amount of future context at the stream start
- For the test set (which starts well into the stream), this is negligible — the affected samples are at the very beginning
- The threshold is tuned on validation labels only, not test labels
- EWMA would be more realistic for streaming but adds an extra hyperparameter; `mode="same"` is a standard simple baseline

---

## D12 — Dataset not committed to git

**Decision:** `data/electricity.csv` is listed in `.gitignore` and not pushed to GitHub.

**Alternatives considered:**
- Commit the dataset directly
- Use Git LFS

**Justification:**
- The file is 3.5 MB — small by data standards but unnecessary to version-control
- The dataset is freely available from datahub.io (link in README)
- Keeping data out of source control is standard ML engineering practice
- Git LFS adds operational complexity for a single CSV file
