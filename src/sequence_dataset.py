"""
Builds the sliding-window sequence dataset used by the LSTM/GRU drift
forecasters.

Input to each window: [C_t, H_t] (confidence, entropy) for the last
`window_len` timesteps.

Label for a window ending at t: "future degradation" = 1 if the model's
rolling error rate over the *next* `horizon` steps is meaningfully higher
than its error rate over the *preceding* `window_len` steps (i.e. the model
is about to get worse), else 0. This operationalises H1/H3 from the
proposal: can confidence/entropy trajectories forecast degradation before
it fully shows up in the error stream.
"""
import numpy as np


def make_windows(confidences: np.ndarray, entropies: np.ndarray, errors: np.ndarray,
                  window_len: int = 20, horizon: int = 20, degradation_delta: float = 0.15):
    """
    Returns:
        X: (N, window_len, 2) windows of [C_t, H_t]
        y: (N,) binary future-degradation labels
        centers: (N,) index t (end of window) for each sample, for alignment
                 with ADWIN/DDM flags in evaluation
    """
    n = len(errors)
    X, y, centers = [], [], []

    err_cumsum = np.concatenate([[0], np.cumsum(errors)])

    def error_rate(a, b):  # mean error over [a, b)
        if b <= a:
            return 0.0
        return (err_cumsum[b] - err_cumsum[a]) / (b - a)

    for t in range(window_len, n - horizon):
        c_win = confidences[t - window_len:t]
        h_win = entropies[t - window_len:t]
        window = np.stack([c_win, h_win], axis=1)  # (window_len, 2)

        past_err = error_rate(t - window_len, t)
        future_err = error_rate(t, t + horizon)
        label = int((future_err - past_err) > degradation_delta)

        X.append(window)
        y.append(label)
        centers.append(t)

    return np.array(X), np.array(y), np.array(centers)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    conf = rng.uniform(0.5, 1.0, n)
    ent = rng.uniform(0.0, 0.7, n)
    err = rng.binomial(1, 0.1, n)
    err[1000:1100] = rng.binomial(1, 0.6, 100)  # injected degradation burst

    X, y, centers = make_windows(conf, ent, err)
    print("windows:", X.shape, "positives:", y.sum(), "/", len(y))
