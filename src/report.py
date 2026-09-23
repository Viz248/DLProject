"""Generate the final outputs from saved artifacts without retraining.

Usage:
    python -m src.report
"""

import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.evaluate import plot_signal_and_alarms, print_final_comparison_table


ARTIFACT_PATH = Path("outputs/pipeline_artifacts.pkl")


def load_artifacts(path: str | Path = ARTIFACT_PATH):
    with Path(path).open("rb") as f:
        artifact = pickle.load(f)
    return artifact


def main():
    artifact = load_artifacts()

    results = artifact["results"]
    drift_summary = artifact["drift_summary"]
    confs = artifact["confs"]
    ents = artifact["ents"]
    errors = artifact["errors"]
    adwin_flags = artifact["adwin_flags"]
    ddm_flags = artifact["ddm_flags"]
    centers_test = artifact["centers_test"]
    forecasts = artifact["forecasts"]
    proxy_drift_index = artifact["proxy_drift_index"]

    print_final_comparison_table(results, drift_summary)

    out_path = plot_signal_and_alarms(
        confs, ents, errors, adwin_flags, ddm_flags,
        lstm_prob=forecasts["lstm"], lstm_centers=centers_test,
        gru_prob=forecasts["gru"], gru_centers=centers_test,
        true_drift_index=proxy_drift_index,
        out_path="outputs/signals.png",
    )
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
