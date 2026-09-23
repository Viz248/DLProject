# Early Detection of Concept Drift and Model Degradation Using Deep Sequence Models

This project studies whether internal model-behavior signals from an early deployed classifier can be used to forecast degradation before it becomes obvious in downstream accuracy metrics.

The pipeline uses the Elec2 electricity stream, trains a baseline MLP classifier, collects confidence, entropy, and error statistics, and then evaluates both statistical drift detectors (ADWIN, DDM) and sequence-based predictors (LSTM and GRU) for warning performance.

## Problem setting

The streaming data is the Australian NSW electricity market dataset (Elec2), with two classes:
- UP
- DOWN

The project is designed around the following idea:
1. Train a baseline classifier on early data.
2. Monitor prediction confidence and entropy as new samples arrive.
3. Build sliding windows from these internal signals.
4. Train an LSTM/GRU to forecast degradation in the baseline model.
5. Compare the warning model against standard drift detectors.

## Current implementation

The project is implemented as a Python pipeline using NumPy, pandas, scikit-learn, and river.

### Main modules
- [src/preprocessing.py](src/preprocessing.py): load and preprocess the Elec2 stream
- [src/mlp.py](src/mlp.py): baseline MLP and streaming evaluation logic
- [src/drift_detectors.py](src/drift_detectors.py): ADWIN and DDM detectors
- [src/sequence_dataset.py](src/sequence_dataset.py): sliding-window dataset generation and proxy degradation labels
- [src/rnn_layers.py](src/rnn_layers.py): custom LSTM and GRU sequence models
- [src/evaluate.py](src/evaluate.py): metrics, delay/FAR calculations, and plotting
- [src/train.py](src/train.py): end-to-end training and artifact generation
- [src/report.py](src/report.py): load saved artifacts and print final table without retraining

## Folder layout

```text
project/
  data/
    electricity.csv
  outputs/
    pipeline_artifacts.pkl
    signals.png
  src/
    __init__.py
    preprocessing.py
    mlp.py
    drift_detectors.py
    sequence_dataset.py
    rnn_layers.py
    evaluate.py
    train.py
    report.py
  requirements.txt
  README.md
```

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

## Run the training pipeline

```bash
python -m src.train
```

This trains the baseline MLP, evaluates streaming confidence and error statistics, runs ADWIN and DDM, trains LSTM and GRU forecasters, and saves the outputs to the outputs folder.

## Regenerate the final table without retraining

Once training has been run and the model artifacts are saved, you can generate the final comparison table and plot from saved data only:

```bash
python -m src.report
```

This is the preferred workflow for report generation because it does not retrain the model each time.

## Outputs

The script writes:
- [outputs/pipeline_artifacts.pkl](outputs/pipeline_artifacts.pkl): saved model and evaluation artifacts
- [outputs/signals.png](outputs/signals.png): plot of confidence, entropy, error, and detector alarms

## Important methodological note

This project uses a proxy drift point rather than a verified ground-truth drift label.

The Elec2 dataset does not contain a known labeled drift event, so the implementation uses a midpoint-based proxy for delay comparison and warning evaluation. This is explicitly documented in the code and report output as an illustrative comparison only, not as a verified drift ground truth.

That means:
- the detection delay values are only meaningful as relative proxy comparisons,
- false-alarm rates are reported under the same proxy setup,
- the results should be interpreted carefully and honestly in the report.

## Current status of the project

The current implementation is working end-to-end and is structured to support a reproducible, report-friendly workflow:
1. train once,
2. save artifacts,
3. load artifacts for final reporting.

The project is intentionally conservative in how it presents results: it does not invent a true drift label or claim a validated drift event that the dataset does not provide.

## Typical usage summary

```bash
cd project
python -m src.train
python -m src.report
```

This gives a full training pass followed by a lightweight report generation step based on saved results.
