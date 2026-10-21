"""
Evaluation metrics and visualisation for the drift-forecasting project.

Metrics
-------
- MLP baseline: accuracy, precision, recall, F1, ROC-AUC, PR-AUC
- LSTM/GRU degradation forecasting: positive-class precision, recall, F1,
  PR-AUC, confusion matrix (threshold tuned on validation)
- ADWIN/DDM: detection rate, mean delay, false-alarm rate, missed detections

All plots saved to reports/figures/.
All numeric results saved to results/.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

FIG_DIR = os.path.join("reports", "figures")
RES_DIR = "results"


def _ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
#  MLP baseline metrics                                                #
# ------------------------------------------------------------------ #
def mlp_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                y_prob: np.ndarray) -> dict:
    """y_prob: probability of class 1 (shape N,)."""
    out = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out["pr_auc"] = float("nan")
    return out


# ------------------------------------------------------------------ #
#  Threshold selection on validation set                               #
# ------------------------------------------------------------------ #
def select_threshold(y_val: np.ndarray, probs_val: np.ndarray,
                     metric: str = "f1") -> float:
    """
    Sweep thresholds on validation probabilities to maximise the chosen
    metric for the positive class.  Returns best threshold in [0, 1].
    """
    best_thresh, best_val = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs_val >= t).astype(int)
        if metric == "f1":
            val = f1_score(y_val, preds, zero_division=0, pos_label=1)
        elif metric == "recall":
            val = recall_score(y_val, preds, zero_division=0, pos_label=1)
        else:
            val = f1_score(y_val, preds, zero_division=0, pos_label=1)
        if val > best_val:
            best_val   = val
            best_thresh = float(t)
    return best_thresh


def select_threshold_precision(y_val: np.ndarray, probs_val: np.ndarray,
                               min_recall: float = 0.25) -> float:
    """
    High-precision operating point: among thresholds whose VALIDATION recall is
    at least `min_recall`, return the one with the highest precision.  Falls
    back to the best-F1 threshold if none qualifies.  (Targeting a precision
    value directly does not transfer well here because the positive rate
    differs between validation and test.)
    """
    best_t, best_prec = None, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs_val >= t).astype(int)
        if preds.sum() < 50:                  # ignore degenerate tiny alarm sets
            continue
        prec = precision_score(y_val, preds, zero_division=0, pos_label=1)
        rec  = recall_score(y_val, preds, zero_division=0, pos_label=1)
        if rec >= min_recall and prec > best_prec:
            best_t, best_prec = float(t), prec
    return best_t if best_t is not None else select_threshold(y_val, probs_val, "f1")


# ------------------------------------------------------------------ #
#  Degradation forecasting metrics                                     #
# ------------------------------------------------------------------ #
def forecasting_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                         threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out = {
        "threshold":        threshold,
        "prevalence":       float(y_true.mean()),
        "precision_pos":    float(precision_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "recall_pos":       float(recall_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "f1_pos":           float(f1_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
    }
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out["pr_auc"] = float("nan")
    return out


# ------------------------------------------------------------------ #
#  Majority-class and rolling-baseline comparators                     #
# ------------------------------------------------------------------ #
def majority_class_metrics(y_true: np.ndarray) -> dict:
    majority = int(np.bincount(y_true).argmax())
    y_pred   = np.full_like(y_true, majority)
    return {
        "threshold":     "majority",
        "prevalence":    float(y_true.mean()),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "recall_pos":    float(recall_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "f1_pos":        float(f1_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "accuracy":      float(accuracy_score(y_true, y_pred)),
        "pr_auc":        float(y_true.mean()),   # constant score -> AP = prevalence
    }


def always_positive_metrics(y_true: np.ndarray) -> dict:
    """Trivial 'always warn' policy: recall 1, precision = prevalence."""
    y_pred = np.ones_like(y_true)
    return {
        "threshold":     "always",
        "prevalence":    float(y_true.mean()),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "recall_pos":    1.0,
        "f1_pos":        float(f1_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "accuracy":      float(accuracy_score(y_true, y_pred)),
        "pr_auc":        float(y_true.mean()),
    }


def _tune_score_threshold(y_val, s_val):
    """Best-F1 threshold for an arbitrary-scale score, searched over validation
    quantiles (never touches test data)."""
    qs = np.unique(np.quantile(s_val, np.linspace(0.40, 0.995, 120)))
    best_t, best_f = float(qs[0]), -1.0
    for t in qs:
        f = f1_score(y_val, (s_val >= t).astype(int), zero_division=0, pos_label=1)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def score_baseline_metrics(s_val: np.ndarray, y_val: np.ndarray,
                           s_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a non-learned baseline given a per-window scalar score where a
    HIGHER score means "more likely degrading".  The decision threshold is
    tuned on validation only, then applied once to the test windows.
    """
    thr    = _tune_score_threshold(y_val, s_val)
    y_pred = (s_test >= thr).astype(int)
    try:
        ap = float(average_precision_score(y_test, s_test))
    except ValueError:
        ap = float("nan")
    return {
        "threshold":     thr,
        "prevalence":    float(y_test.mean()),
        "precision_pos": float(precision_score(y_test, y_pred, zero_division=0, pos_label=1)),
        "recall_pos":    float(recall_score(y_test, y_pred, zero_division=0, pos_label=1)),
        "f1_pos":        float(f1_score(y_test, y_pred, zero_division=0, pos_label=1)),
        "accuracy":      float(accuracy_score(y_test, y_pred)),
        "pr_auc":        ap,
    }


# ------------------------------------------------------------------ #
#  Save helpers                                                        #
# ------------------------------------------------------------------ #
def save_json(data: dict, filename: str):
    _ensure_dirs()
    path = os.path.join(RES_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def save_numpy(arr: np.ndarray, filename: str):
    _ensure_dirs()
    path = os.path.join(RES_DIR, filename)
    np.save(path, arr)
    return path


# ------------------------------------------------------------------ #
#  Plots                                                               #
# ------------------------------------------------------------------ #
def plot_mlp_over_time(errors, confs, ents, out_name="mlp_signals.png"):
    _ensure_dirs()
    win = 200
    roll_err = np.convolve(errors, np.ones(win) / win, mode="same")
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(roll_err, color="tab:red", linewidth=1)
    axes[0].set_ylabel("Rolling error rate\n(window=200)")
    axes[0].set_title("MLP prequential performance over the Elec2 stream")
    axes[0].set_ylim(0, 1)

    axes[1].plot(confs, color="tab:blue", linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel("Confidence C_t")
    axes[1].set_ylim(0, 1)

    axes[2].plot(ents, color="tab:green", linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel("Entropy H_t")

    axes[-1].set_xlabel("Stream index t (relative to warm-up end)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_signals_and_detectors(confs, ents, errors, adwin_flags, ddm_flags,
                                 lstm_prob=None, lstm_centers=None,
                                 gru_prob=None,  gru_centers=None,
                                 out_name="signals_detectors.png"):
    _ensure_dirs()
    win = 100
    roll_err = np.convolve(errors, np.ones(win) / win, mode="same")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(confs, label="Confidence C_t", alpha=0.6, linewidth=0.6)
    axes[0].plot(ents,  label="Entropy H_t",    alpha=0.6, linewidth=0.6)
    axes[0].set_ylabel("Signal");  axes[0].legend(loc="upper right")
    axes[0].set_title("MLP internal monitoring signals")

    axes[1].plot(roll_err, color="black", linewidth=1,
                 label=f"Rolling error (w={win})")
    for idx in np.where(adwin_flags)[0]:
        axes[1].axvline(idx, color="tab:orange", alpha=0.25, linewidth=0.6)
    for idx in np.where(ddm_flags)[0]:
        axes[1].axvline(idx, color="tab:red",    alpha=0.25, linewidth=0.6,
                        linestyle="--")
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[
        Line2D([0],[0], color="black",      label=f"Rolling error (w={win})"),
        Line2D([0],[0], color="tab:orange", label="ADWIN alarm"),
        Line2D([0],[0], color="tab:red",    label="DDM alarm", linestyle="--"),
    ], loc="upper right", fontsize=8)
    axes[1].set_ylabel("Error rate")
    axes[1].set_title("Error stream with ADWIN / DDM alarms")

    if lstm_prob is not None:
        axes[2].plot(lstm_centers, lstm_prob,
                     label="LSTM P(degradation)", color="tab:blue", linewidth=0.8)
    if gru_prob is not None:
        axes[2].plot(gru_centers, gru_prob,
                     label="GRU P(degradation)", color="tab:green", linewidth=0.8)
    axes[2].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[2].set_ylabel("P(future degrad.)")
    axes[2].set_title("Sequence-model degradation forecasts")
    axes[2].legend(loc="upper right")

    axes[-1].set_xlabel("Stream index t (relative to warm-up end)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm: np.ndarray, model_name: str,
                           out_name: str = None):
    _ensure_dirs()
    if out_name is None:
        out_name = f"cm_{model_name.lower()}.png"
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray,
                  model_name: str, out_name: str = None):
    _ensure_dirs()
    if out_name is None:
        out_name = f"pr_{model_name.lower()}.png"
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, linewidth=2, label=f"AP={ap:.3f}")
    ax.axhline(y_true.mean(), color="gray", linestyle="--",
               label=f"Baseline (prevalence={y_true.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_degradation_probs(centers, probs_lstm, probs_gru, labels,
                            out_name="degradation_probs.png"):
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(centers, probs_lstm, label="LSTM", alpha=0.8, linewidth=0.9,
            color="tab:blue")
    ax.plot(centers, probs_gru,  label="GRU",  alpha=0.8, linewidth=0.9,
            color="tab:green")
    ax.scatter(centers[labels == 1], np.full(labels.sum(), -0.03),
               marker="|", s=40, color="tab:red", label="True degradation",
               zorder=5)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Stream index"); ax.set_ylabel("P(future degradation)")
    ax.set_title("Degradation probabilities over the test stream")
    ax.legend(); ax.set_ylim(-0.1, 1.05)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_synthetic_drift_experiment(errors, change_points,
                                     adwin_flags, ddm_flags,
                                     out_name="synthetic_drift.png"):
    _ensure_dirs()
    win = 50
    roll_err = np.convolve(errors, np.ones(win) / win, mode="same")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(roll_err, color="black", linewidth=1.2, label=f"Rolling error (w={win})")
    for cp in change_points:
        ax.axvline(cp, color="purple", linewidth=2, linestyle="-",
                   label="True change point")
    for idx in np.where(adwin_flags)[0]:
        ax.axvline(idx, color="tab:orange", alpha=0.4, linewidth=0.8)
    for idx in np.where(ddm_flags)[0]:
        ax.axvline(idx, color="tab:red", alpha=0.4, linewidth=0.8,
                   linestyle="--")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color="black",      label=f"Rolling error (w={win})"),
        Line2D([0],[0], color="purple",     label="True change point"),
        Line2D([0],[0], color="tab:orange", label="ADWIN alarm"),
        Line2D([0],[0], color="tab:red",    label="DDM alarm", linestyle="--"),
    ], loc="upper left", fontsize=9)
    ax.set_xlabel("Stream index"); ax.set_ylabel("Error rate")
    ax.set_title("Synthetic drift experiment — known change point vs detector alarms")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_model_comparison(results_table: dict, out_name="model_comparison.png"):
    """
    results_table: {model_name: {f1_pos, pr_auc, recall_pos, precision_pos}}
    """
    _ensure_dirs()
    names   = list(results_table.keys())
    metrics = ["f1_pos", "pr_auc", "recall_pos", "precision_pos"]
    labels  = ["F1 (pos)", "PR-AUC", "Recall (pos)", "Precision (pos)"]
    x = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (m, lbl) in enumerate(zip(metrics, labels)):
        vals = [results_table[n].get(m, 0) or 0 for n in names]
        ax.bar(x + k * width, vals, width, label=lbl)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Degradation forecasting model comparison")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ablation(ablation_results: dict, out_name="ablation.png"):
    _ensure_dirs()
    names = list(ablation_results.keys())
    f1s   = [ablation_results[n].get("f1_pos", 0) or 0 for n in names]
    prs   = [ablation_results[n].get("pr_auc", 0) or 0 for n in names]

    x     = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, f1s, width, label="F1 (pos)")
    ax.bar(x + width/2, prs, width, label="PR-AUC")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Monitoring-signal ablation study")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_table_a(mlp_m: dict):
    print("\n=== TABLE A — Baseline MLP Classification ===")
    print(f"{'Metric':<14} {'Value':>8}")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        v = mlp_m.get(k, float("nan"))
        print(f"  {k:<12} {v:>8.4f}")


def print_table_b(fcast_results: dict):
    print("\n=== TABLE B — Degradation Forecasting ===")
    hdr = f"{'Model':<14}{'Prec':>8}{'Rec':>8}{'F1':>8}{'PR-AUC':>9}{'Acc':>8}{'Thr':>7}{'Prev':>8}"
    print(hdr)
    for name, m in fcast_results.items():
        thr  = m.get("threshold", float("nan"))
        thr_str = f"{thr:.2f}" if isinstance(thr, float) else str(thr)
        print(f"  {name:<12}"
              f"{m.get('precision_pos', float('nan')):>8.3f}"
              f"{m.get('recall_pos', float('nan')):>8.3f}"
              f"{m.get('f1_pos', float('nan')):>8.3f}"
              f"{m.get('pr_auc', float('nan')):>9.3f}"
              f"{m.get('accuracy', float('nan')):>8.3f}"
              f"{thr_str:>7}"
              f"{m.get('prevalence', float('nan')):>8.3f}")


def print_table_c(det_results: dict):
    print("\n=== TABLE C — Drift Detection (Synthetic Streams) ===")
    hdr = f"{'Detector':<10}{'Alarms':>8}{'Detected':>10}{'Missed':>8}"
    hdr += f"{'FA':>6}{'MeanDelay':>11}{'DetRate':>9}{'FARate':>8}"
    print(hdr)
    for name, m in det_results.items():
        md = m.get("mean_delay")
        md_str = f"{md:.1f}" if md is not None else "N/A"
        print(f"  {name:<8}"
              f"{m.get('n_alarms', 0):>8}"
              f"{m.get('n_true_detections', 0):>10}"
              f"{m.get('n_missed', 0):>8}"
              f"{m.get('n_false_alarms', 0):>6}"
              f"{md_str:>11}"
              f"{m.get('detection_rate', float('nan')):>9.3f}"
              f"{m.get('false_alarm_rate', float('nan')):>8.3f}")


def print_table_d(ew: dict):
    print("\n=== TABLE D - Early warning vs reactive detectors (test stream) ===")
    hdr = (f"{'Method':<16}{'Alarms':>7}{'Events':>7}{'Recall':>8}{'Chance':>8}"
           f"{'Early':>7}{'MedOff':>8}{'AlPrec':>8}{'FA/1000':>9}")
    print(hdr)
    for name, m in ew.items():
        mo = m.get("median_offset")
        mo_s = f"{mo:.0f}" if mo is not None else "N/A"
        print(f"  {name:<14}{m['n_alarms']:>7}{m['n_events']:>7}"
              f"{m['event_recall']:>8.3f}{m['chance_event_recall']:>8.3f}"
              f"{m['early_rate']:>7.3f}{mo_s:>8}"
              f"{m['alarm_precision']:>8.3f}{m['false_alarms_per_1000']:>9.1f}")


def plot_early_warning(ew: dict, out_name: str = "early_warning.png"):
    _ensure_dirs()
    names = list(ew.keys())
    rec   = [ew[n]["event_recall"] for n in names]
    cha   = [ew[n]["chance_event_recall"] for n in names]
    fa    = [ew[n]["false_alarms_per_1000"] for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(x - w/2, rec, w, label="Event recall")
    axes[0].bar(x + w/2, cha, w, label="Random alarms (same count)",
                color="lightgray")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Fraction of degradation events hit")
    axes[0].set_title("Hit rate vs chance"); axes[0].legend()
    axes[1].bar(x, fa, color="tab:red", alpha=0.7)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].set_ylabel("Alarms matching no event / 1000 steps")
    axes[1].set_title("False-alarm load")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path
