"""
Print the final result tables from saved JSON (no retraining).

Usage:
    python -m src.report
"""
import json
import os
import sys

RES_DIR = "results"


def _load(name):
    path = os.path.join(RES_DIR, name)
    if not os.path.exists(path):
        sys.exit(f"Missing {path} - run `python -m src.train` first.")
    with open(path) as f:
        return json.load(f)


def main():
    from src.evaluate import print_table_a, print_table_b, print_table_d
    meta = _load("experiment_metadata.json")
    print(f"Windows: train={meta['n_train_windows']}  val={meta['n_val_windows']}  "
          f"test={meta['n_test_windows']}   positive rate "
          f"train/val/test = {meta['pos_rate']['train']:.3f}/"
          f"{meta['pos_rate']['val']:.3f}/{meta['pos_rate']['test']:.3f}")
    print_table_a(_load("table_a_mlp_metrics.json"))
    print_table_b(_load("table_b_forecasting.json"))
    print("\n=== Ablation (LSTM) ===")
    for k, v in _load("ablation_results.json").items():
        print(f"  {k:<10} F1={v['f1_pos']:.3f} +-{v['f1_std']:.3f}  "
              f"PR-AUC={v['pr_auc']:.3f} +-{v['pr_std']:.3f}")
    print_table_d(_load("table_d_early_warning.json"))


if __name__ == "__main__":
    main()
