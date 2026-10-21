# Novelty: RiskGate-CUSUM (forecast-gated sequential change detection)

## What is new
The base project compared two families of methods: error-based detectors (ADWIN, DDM) that are **reliable but late**,
and LSTM/GRU forecasters that read the classifier's confidence/entropy and are **early but noisy**.
**RiskGate-CUSUM** fuses them *inside* one sequential test instead of voting between two separate alarms.
A GRU forecaster's risk score is turned into a scale-free, causal percentile rank `u_t` in [0, 1]
(rank among the forecaster's own previous 2,000 outputs) and then drives a one-sided CUSUM on the error stream in two ways:

```
S_t = max(0, S_(t-1) + (e_t - mu_t - k) + gamma * max(0, u_t - u0))     (1) evidence injection
alarm when  S_t > h_t ,   h_t = h0 * (1 - beta * u_t)                   (2) threshold gating
```

* `e_t` error indicator, `mu_t` causal reference error rate (mean of the last 500 errors), `k` allowance.
* (1) risk above the priming level `u0` adds evidence before errors accumulate.
* (2) high risk lowers the threshold; low risk leaves it at `h0`, so the detector is sensitive mainly when trouble is likely.
* Special cases: `beta = gamma = 0` is the plain CUSUM, `gamma = 0` is gate-only, `beta = 0` is inject-only.
* Everything is strictly causal and online (`RiskGateCUSUM.update(error, risk)`). A cold-start guard keeps the statistic at 0
  until 100 past errors exist (without it the reference rate starts at 0 and the test fires spuriously).

**Novelty claim (read this).** CUSUM, drift forecasting and label-free confidence-based detectors are all known
(for example Sethi and Kantardzic, "On the reliable detection of concept drift from streaming unlabeled data", arXiv:1704.00023,
and work that uses 1 - max-probability as a surrogate error signal with CUSUM/Page-Hinkley). The contribution here is the
specific coupling: a neural forecaster's rank-normalised risk driving **both the evidence term and the threshold** of the error CUSUM.
A quick search found no paper with this exact combination, but that is not a proof of novelty. Do a proper literature search
before claiming it is new in a report or patent.

## Files
| File | Purpose |
|---|---|
| `src/riskgate.py` | the algorithm (`RiskGateParams`, `RiskRank`, `RiskGateCUSUM`, `plain_cusum`) |
| `src/novelty_experiment.py` | full experiment: train GRU forecaster, tune on validation, evaluate once on test |
| `tests/test_riskgate.py` | 24 tests (causality, reductions to plain CUSUM, gating, injection, cold start, tuning) |
| `docs/novelty/` | results JSON and figure from the run below |

## Run
```bash
pip install -r requirements.txt
python -m pytest tests/ -q                       # 74 tests
python -m src.novelty_experiment --quick         # ~30 s smoke run
python -m src.novelty_experiment --seeds 1       # ~4-8 min, one forecaster seed
python -m src.novelty_experiment                 # 3 forecaster seeds (slow on one core)
```

## Protocol (no test leakage)
Same stream, MLP warm-up, causal scaling and window split as the main pipeline. A GRU forecaster is trained per seed.
Each detector family (plain, gate-only, inject-only, RiskGate) is tuned **on the validation stream only** by grid search
(h0, k, beta, gamma): maximise event-level F1 subject to at most 6 alarms per 1,000 steps. The tuned detectors are then run
**once** on the test stream and compared on the same 25 severe-degradation events as Table D of the report
(hit = alarm within 30 steps before to 60 steps after the onset).

## Results (test stream, 1 forecaster seed, 25 events, 8097 steps)
Tuned RiskGate parameters: h0 = 6.0, k = 0.1, beta = 0.3, gamma = 1.0, u0 = 0.8.

| Method | Alarms | Event recall | Chance recall | Alarm precision | Event F1 | Warned before onset | Median offset (steps) | False alarms / 1000 |
|---|---|---|---|---|---|---|---|---|
| CUSUM (plain) | 26 | 0.80 | 0.25 | 0.88 | 0.840 | 0.00 | 21 | 0.4 |
| Gate-only | 22 | 0.76 | 0.21 | 0.95 | 0.846 | 0.00 | 20 | 0.1 |
| Inject-only | 27 | 0.84 | 0.26 | 0.89 | 0.864 | 0.00 | 20 | 0.4 |
| RiskGate-CUSUM | 25 | 0.80 | 0.24 | 0.96 | 0.873 | 0.00 | 19 | 0.1 |
| Forecaster only | 63 | 0.60 | 0.50 | 0.30 | 0.401 | 0.36 | -6 | 5.4 |
| Naive OR (CUSUM+Forecaster) | 83 | 0.84 | 0.60 | 0.46 | 0.593 | 0.36 | 7 | 5.6 |
| ADWIN | 1 | 0.04 | 0.01 | 1.00 | 0.077 | 0.04 | -28 | 0.0 |
| ADWIN(d=.05) | 10 | 0.24 | 0.11 | 0.60 | 0.343 | 0.04 | 38 | 0.5 |
| DDM | 0 | 0.00 | 0.00 | n/a | 0.000 | 0.00 | n/a | 0.0 |

## Honest reading
* **Working code, small effect.** RiskGate-CUSUM has the best event F1 (0.873 vs 0.840 for plain CUSUM),
  mostly through higher alarm precision and fewer false alarms. **The difference is about one alarm** (24 of 25 alarms hit versus 23 of 26), with one
  forecaster seed and 25 events, so it is **not statistically meaningful**. Do not present it as a proven improvement.
* **It does not give earlier warning.** None of the CUSUM variants alarm before the onset (median offset about +20 steps); they use errors, which only exist
  after degradation starts. Only the forecaster alone warns early (median offset -6), at low precision.
* **Fusion helps over naive voting.** Fusing inside the test is clearly better than the naive "CUSUM OR forecaster" union
  (F1 0.593), which inherits the forecaster's many false alarms.
* **Unfair to ADWIN/DDM.** The CUSUM variants were tuned on validation; ADWIN and DDM use default settings and were not tuned.
  The event definition is also based on error rates, which favours error-based detectors. Their low scores here are not a general verdict.
* **Next steps for a real claim:** run `--seeds 3` or more, evaluate on additional drift datasets, tune ADWIN/DDM on validation as well,
  add a significance test over many events, and consider events defined independently of the error stream.
