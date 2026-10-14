"""
Sliding-window sequence dataset for the LSTM/GRU degradation forecasters.

Mathematical definition of the degradation label
-------------------------------------------------
Let err[t] be the binary error indicator at stream position t.

For a window ending at position t (exclusive), the past error rate is:
    past_err(t)   = mean(err[t - window_len : t])

The future error rate at horizon h is:
    future_err(t) = mean(err[t : t + horizon])

The degradation label is:
    label(t) = 1  if  future_err(t) - past_err(t) > degradation_delta
               0  otherwise

The INPUT to the model is the sequence of [C_s, H_s] pairs for
s in [t - window_len, t) — strictly PAST information at prediction time t.

The TARGET uses future error rates (err[t : t + horizon]) which are unknown
at prediction time; they are only used to construct the training label
(ground truth) and do NOT appear in the model input.

Key invariant: the window boundary index `t` is the prediction timestamp;
any information at index >= t is future and must never enter the input.
"""
from typing import Optional
import numpy as np


def make_windows(confidences: np.ndarray,
                 entropies: np.ndarray,
                 errors: np.ndarray,
                 window_len: int = 20,
                 horizon: int = 20,
                 degradation_delta: float = 0.15,
                 extra_features: Optional[np.ndarray] = None):
    """
    Build the sequence dataset.

    Parameters
    ----------
    confidences, entropies, errors : 1-D arrays of length N,
        output of run_stream() — these all share the same index space.
    extra_features : optional (N, F) array of additional input features.
        If provided, each timestep input becomes [C_t, H_t, *extra_t].
    window_len, horizon, degradation_delta : see module docstring.

    Returns
    -------
    X       : (M, window_len, n_features)   — sequence inputs
    y       : (M,)                          — binary degradation labels
    centers : (M,)                          — stream index t (end of each window)
    past_err_arr, future_err_arr : (M,) — stored for transparency / debugging
    """
    n = len(errors)
    assert len(confidences) == n and len(entropies) == n, "Array length mismatch"

    if extra_features is not None:
        assert extra_features.shape[0] == n, "extra_features length mismatch"

    err_cumsum = np.concatenate([[0], np.cumsum(errors)])

    def error_rate(a, b):
        if b <= a:
            return 0.0
        return float(err_cumsum[b] - err_cumsum[a]) / (b - a)

    X_list, y_list, centers_list = [], [], []
    past_err_list, future_err_list = [], []

    # t is the END of the window (first future index)
    for t in range(window_len, n - horizon):
        # --- inputs: strictly past ---
        c_win = confidences[t - window_len : t]
        h_win = entropies[t - window_len : t]

        if extra_features is not None:
            e_win = extra_features[t - window_len : t]   # (window_len, F)
            window = np.concatenate(
                [np.stack([c_win, h_win], axis=1), e_win], axis=1)
        else:
            window = np.stack([c_win, h_win], axis=1)    # (window_len, 2)

        # --- label: uses future info — never enters model input ---
        past_err   = error_rate(t - window_len, t)
        future_err = error_rate(t, t + horizon)
        label      = int((future_err - past_err) > degradation_delta)

        X_list.append(window)
        y_list.append(label)
        centers_list.append(t)
        past_err_list.append(past_err)
        future_err_list.append(future_err)

    return (np.array(X_list),
            np.array(y_list),
            np.array(centers_list),
            np.array(past_err_list),
            np.array(future_err_list))


def split_windows(X, y, centers, past_err, future_err,
                  val_start_abs, test_start_abs):
    """
    Split pre-built windows into train / val / test using absolute stream
    indices stored in `centers`.  Windows whose centre falls inside the
    temporal gap (between splits) are simply excluded — no boundary leakage.
    """
    tr  = centers < val_start_abs
    val = (centers >= val_start_abs) & (centers < test_start_abs)
    te  = centers >= test_start_abs

    def _sel(mask):
        return X[mask], y[mask], centers[mask], past_err[mask], future_err[mask]

    return _sel(tr), _sel(val), _sel(te)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    conf = rng.uniform(0.5, 1.0, n)
    ent  = rng.uniform(0.0, 0.7, n)
    err  = rng.binomial(1, 0.1, n)
    err[1000:1100] = rng.binomial(1, 0.6, 100)

    X, y, centers, pe, fe = make_windows(conf, ent, err)
    print(f"Windows: {X.shape}, positives: {y.sum()}/{len(y)} "
          f"({y.mean():.2%})")
