"""
Evaluation metrics for the drift-forecasting models (LSTM/GRU) and the
statistical baselines (ADWIN/DDM), plus comparison plots.

Metrics:
    - Precision, Recall, F1, Accuracy, AUC  (for LSTM/GRU probabilistic outputs
      against the future-degradation label)
    - Detection delay: for a known injected-drift point, how many samples
      after the true drift point did each method raise its first alarm
    - False alarm rate: fraction of alarms raised outside any true-drift window
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["auc"] = float("nan")
    return out


def detection_delay(flag_indices: np.ndarray, true_drift_index: int):
    """flag_indices: sorted array of indices (in the same coordinate system
    as true_drift_index) where a method raised an alarm.

    Returns the delay in samples to the first alarm at or after the drift point.
    If the drift point is only a proxy/illustrative marker rather than a verified
    ground-truth drift event, the caller should label the result as such.
    """
    after = flag_indices[flag_indices >= true_drift_index]
    if len(after) == 0:
        return None
    return int(after[0] - true_drift_index)


def false_alarm_rate(flag_indices: np.ndarray, true_drift_windows: list, total_len: int):
    """true_drift_windows: list of (start, end) tuples marking legitimate
    drift periods. Any alarm outside all windows counts as a false alarm.

    This metric is only meaningful when the drift windows represent real, known
    drift labels. If the windows are only illustrative proxy regions, the
    caller should treat the resulting rate as a proxy evaluation rather than a
    fully verified measure.
    """
    if len(flag_indices) == 0:
        return 0.0

    def in_any_window(idx):
        return any(s <= idx <= e for s, e in true_drift_windows)

    false_alarms = sum(1 for idx in flag_indices if not in_any_window(idx))
    return false_alarms / len(flag_indices)


def plot_signal_and_alarms(confs, ents, errors, adwin_flags, ddm_flags,
                            lstm_prob=None, lstm_centers=None,
                            gru_prob=None, gru_centers=None,
                            true_drift_index=None, out_path="outputs/signals.png"):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(confs, label="Confidence C_t", alpha=0.7)
    axes[0].plot(ents, label="Entropy H_t", alpha=0.7)
    axes[0].set_ylabel("Signal value")
    axes[0].legend(loc="upper right")
    axes[0].set_title("MLP internal signals")

    axes[1].plot(np.convolve(errors, np.ones(50) / 50, mode="same"),
                 label="Rolling error rate (window=50)", color="black")
    for idx in np.where(adwin_flags)[0]:
        axes[1].axvline(idx, color="tab:orange", alpha=0.3)
    for idx in np.where(ddm_flags)[0]:
        axes[1].axvline(idx, color="tab:red", alpha=0.3, linestyle="--")
    axes[1].set_ylabel("Error rate")
    axes[1].set_title("Error stream with ADWIN (orange) / DDM (red) alarms")
    axes[1].legend(loc="upper right")

    if lstm_prob is not None:
        axes[2].plot(lstm_centers, lstm_prob, label="LSTM P(degradation)", color="tab:blue")
    if gru_prob is not None:
        axes[2].plot(gru_centers, gru_prob, label="GRU P(degradation)", color="tab:green")
    axes[2].axhline(0.5, color="gray", linestyle=":")
    axes[2].set_ylabel("P(future degradation)")
    axes[2].set_title("Proposed sequence-model forecasts")
    axes[2].legend(loc="upper right")

    if true_drift_index is not None:
        for ax in axes:
            ax.axvline(true_drift_index, color="purple", linewidth=2, label="True drift")

    axes[-1].set_xlabel("Stream index t")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def print_comparison_table(results: dict):
    print(f"\n{'Model':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Accuracy':>10}{'AUC':>10}")
    for name, m in results.items():
        print(f"{name:<12}{m['precision']:>10.3f}{m['recall']:>10.3f}"
              f"{m['f1']:>10.3f}{m['accuracy']:>10.3f}{m['auc']:>10.3f}")
