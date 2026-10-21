"""
Statistical, error-based concept drift baselines: ADWIN and DDM from `river`.

Fix vs original: DDM was imported from river.drift.binary (correct path).
Added run_detectors_on_stream() for unified benchmarking interface.
"""
import time
import numpy as np
from river.drift import ADWIN
from river.drift.binary import DDM


def run_adwin(errors: np.ndarray, delta: float = 0.002):
    """Returns boolean flag array (True = drift alarm at that index)."""
    detector = ADWIN(delta=delta)
    flags = np.zeros(len(errors), dtype=bool)
    for i, e in enumerate(errors):
        detector.update(int(e))
        if detector.drift_detected:
            flags[i] = True
    return flags


def run_ddm(errors: np.ndarray):
    """Returns (drift_flags, warning_flags) boolean arrays."""
    detector = DDM()
    flags    = np.zeros(len(errors), dtype=bool)
    warnings = np.zeros(len(errors), dtype=bool)
    for i, e in enumerate(errors):
        detector.update(int(e))
        if detector.drift_detected:
            flags[i] = True
        if getattr(detector, "warning_detected", False):
            warnings[i] = True
    return flags, warnings


def run_detectors_on_stream(errors: np.ndarray, adwin_delta: float = 0.002):
    """
    Run both detectors and return a dict with flags + runtime (seconds).
    """
    t0 = time.perf_counter()
    adwin_flags = run_adwin(errors, delta=adwin_delta)
    adwin_time  = time.perf_counter() - t0

    t0 = time.perf_counter()
    ddm_flags, ddm_warn = run_ddm(errors)
    ddm_time = time.perf_counter() - t0

    return {
        "adwin": {"flags": adwin_flags, "runtime_s": adwin_time},
        "ddm":   {"flags": ddm_flags,   "warnings": ddm_warn,
                  "runtime_s": ddm_time},
    }


# --------------------------------------------------------------------------- #
#  Controlled synthetic-stream benchmark helpers                               #
# --------------------------------------------------------------------------- #
def make_synthetic_stream(n_before: int = 2000, n_after: int = 2000,
                           p_before: float = 0.10, p_after: float = 0.45,
                           seed: int = 42):
    """
    Create a binary error stream with one known change point.
    Returns (errors, change_point_idx).
    The true change point is `n_before` (first index with changed distribution).
    """
    rng = np.random.default_rng(seed)
    before = rng.binomial(1, p_before, n_before)
    after  = rng.binomial(1, p_after,  n_after)
    errors = np.concatenate([before, after])
    return errors, n_before


def make_no_change_stream(n: int = 4000, p: float = 0.10, seed: int = 42):
    """Stationary stream for false-alarm baseline."""
    rng = np.random.default_rng(seed)
    return rng.binomial(1, p, n)


def detector_metrics(flags: np.ndarray, change_points: list,
                     grace: int = 0, total_len: int = None):
    """
    Compute detection metrics for one detector on one stream.

    Parameters
    ----------
    flags        : boolean alarm array (length = stream length)
    change_points: list of known change-point indices (first index of new dist.)
    grace        : samples before a change point that still count as true alarm
                   (0 = strict; a small value allows pre-detection)
    total_len    : stream length (defaults to len(flags))

    Returns
    -------
    dict with: n_alarms, n_true_detections, n_missed, n_false_alarms,
               detection_delays (list), detection_rate, false_alarm_rate
    """
    if total_len is None:
        total_len = len(flags)

    alarm_indices = np.where(flags)[0].tolist()

    # Build "valid detection windows" around each change point.
    # An alarm is a true detection if it falls in [cp - grace, stream_end).
    # Once a change point is "claimed" by the first alarm after it, we
    # don't double-count it.
    cp_detected = [False] * len(change_points)
    delays = []

    # Track which alarms are true positives
    true_alarm_set = set()

    for alarm in alarm_indices:
        for k, cp in enumerate(change_points):
            window_start = cp - grace
            if not cp_detected[k] and alarm >= window_start:
                cp_detected[k] = True
                delays.append(max(0, alarm - cp))
                true_alarm_set.add(alarm)
                break   # one alarm claims at most one change point

    n_true_det  = sum(cp_detected)
    n_missed    = len(change_points) - n_true_det
    n_false     = len(alarm_indices) - len(true_alarm_set)
    dr          = n_true_det / len(change_points) if change_points else float("nan")
    far_denom   = len(alarm_indices) if alarm_indices else 1
    far         = n_false / far_denom if alarm_indices else 0.0

    return {
        "n_alarms":           len(alarm_indices),
        "n_true_detections":  n_true_det,
        "n_missed":           n_missed,
        "n_false_alarms":     n_false,
        "detection_delays":   delays,
        "mean_delay":         float(np.mean(delays)) if delays else None,
        "detection_rate":     dr,
        "false_alarm_rate":   far,
    }


if __name__ == "__main__":
    errors, cp = make_synthetic_stream()
    adwin_f  = run_adwin(errors)
    ddm_f, _ = run_ddm(errors)
    for name, f in [("ADWIN", adwin_f), ("DDM", ddm_f)]:
        m = detector_metrics(f, [cp], total_len=len(errors))
        print(f"{name}: {m}")
