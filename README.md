# Internal-Signal Concept Drift Forecasting on Elec2

**Research question:** Can internal prediction signals (confidence, entropy)
from a deployed MLP be used by sequence models (LSTM/GRU) to forecast model
degradation *before* conventional error-based drift detectors (ADWIN, DDM)?

## Structure
```
project/
  data/electricity.csv          # Elec2 dataset
  src/
    preprocessing.py            # load + scale the stream
    mlp.py                      # numpy MLP base classifier, prequential runner
    drift_detectors.py          # ADWIN / DDM wrappers (river)
    rnn_layers.py                # from-scratch LSTM + GRU cells, SequenceHead
    gradcheck.py                 # numerical gradient check for LSTM/GRU
    sequence_dataset.py         # sliding-window dataset + degradation labels
    evaluate.py                  # metrics, detection delay, plots
    train.py                     # end-to-end orchestration
  requirements.txt
```

## Run
```bash
pip install -r requirements.txt
cd project
python -m src.gradcheck     # verify LSTM/GRU backward passes
python -m src.train          # full pipeline: MLP -> ADWIN/DDM -> LSTM/GRU -> eval + plot
```
Output plot and metrics table are written to `outputs/`.

## Notes / next steps for the report
- `torch` could not be installed in this environment (no free disk /
  no CUDA-free wheel reachable), so the MLP, LSTM and GRU are all
  implemented directly in numpy with manual backprop — this is actually a
  plus for the viva, since you can walk through the exact gradient math.
- The "future degradation" label used to train LSTM/GRU is currently a
  placeholder rule (rolling error rate rises by >0.15 over the next 20
  steps vs. the previous 20). Tune `degradation_delta` / `horizon` in
  `sequence_dataset.py` against your actual injected/natural drift points.
- `train.py` currently marks the stream midpoint as a placeholder
  "true drift index" for the detection-delay comparison — replace with your
  actual injected-drift index for a clean ADWIN/DDM vs LSTM/GRU delay comparison.
- Current run shows low LSTM/GRU recall (predicts majority class) — with
  only ~21% positive windows, add class weighting or oversampling in
  `train.py`'s training loop before the demo if you want more balanced
  precision/recall to report.
