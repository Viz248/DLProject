"""
Statistical, error-based concept drift baselines: ADWIN and DDM, both from
`river`. These react to the MLP's binary error stream E_t and serve as the
"conventional" baselines that the proposed LSTM/GRU internal-signal model
is compared against.
"""
import numpy as np
from river.drift import ADWIN
from river.drift.binary import DDM


def run_adwin(errors: np.ndarray, delta: float = 0.002):
    """Returns a boolean array, True at indices where ADWIN signals drift."""
    detector = ADWIN(delta=delta)
    flags = np.zeros(len(errors), dtype=bool)
    for i, e in enumerate(errors):
        detector.update(int(e))
        if detector.drift_detected:
            flags[i] = True
    return flags


def run_ddm(errors: np.ndarray):
    """Returns a boolean array, True at indices where DDM signals a drift
    (warning-level DDM state is also available for finer-grained analysis)."""
    detector = DDM()
    flags = np.zeros(len(errors), dtype=bool)
    warnings = np.zeros(len(errors), dtype=bool)
    for i, e in enumerate(errors):
        detector.update(int(e))
        if detector.drift_detected:
            flags[i] = True
        if getattr(detector, "warning_detected", False):
            warnings[i] = True
    return flags, warnings


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    errs = np.concatenate([
        rng.binomial(1, 0.1, 1000),
        rng.binomial(1, 0.5, 1000),  # injected drift
    ])
    adwin_flags = run_adwin(errs)
    ddm_flags, ddm_warn = run_ddm(errs)
    print("ADWIN first detection at:", np.argmax(adwin_flags) if adwin_flags.any() else None)
    print("DDM first detection at:", np.argmax(ddm_flags) if ddm_flags.any() else None)
