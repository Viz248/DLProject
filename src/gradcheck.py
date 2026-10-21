"""
Numerical gradient check for LSTM and GRU cells + SequenceHead.
Run:  python -m src.gradcheck
"""
import numpy as np
from src.rnn_layers import RNNDriftDetector


def numerical_grad_check(cell_type: str, T: int = 5, n_in: int = 2,
                          n_hidden: int = 4, eps: float = 1e-5,
                          tol: float = 2e-2, seed: int = 0,
                          n_sample: int = 8):
    rng   = np.random.default_rng(seed)
    model = RNNDriftDetector(cell_type=cell_type, n_in=n_in,
                              n_hidden=n_hidden, seed=seed)
    window = rng.normal(size=(T, n_in))
    label  = 1

    # Analytic gradients
    h, cache = model.cell.forward(window)
    p        = model.head.forward(h)
    dh, dW_head, db_head = model.head.backward(h, p, label)
    cell_grads = model.cell.backward(dh, cache)

    def loss_fn():
        h_, _ = model.cell.forward(window)
        p_    = model.head.forward(h_)
        return -(label * np.log(p_ + 1e-12) + (1 - label) * np.log(1 - p_ + 1e-12))

    max_rel_err = 0.0

    for name, analytic_grad in cell_grads.items():
        param = getattr(model.cell, name)
        it = np.nditer(param, flags=["multi_index"])
        checked = 0
        while not it.finished and checked < n_sample:
            idx  = it.multi_index
            orig = param[idx]

            param[idx] = orig + eps
            lp = loss_fn()
            param[idx] = orig - eps
            lm = loss_fn()
            param[idx] = orig

            numeric  = (lp - lm) / (2 * eps)
            analytic = analytic_grad[idx]
            denom    = max(abs(numeric), abs(analytic), 1e-8)
            rel_err  = abs(numeric - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)
            checked += 1
            it.iternext()

    # Also check head weights
    orig_W = model.head.W.copy()
    for j in range(min(n_sample, len(orig_W))):
        model.head.W[j] = orig_W[j] + eps
        lp = loss_fn()
        model.head.W[j] = orig_W[j] - eps
        lm = loss_fn()
        model.head.W[j] = orig_W[j]
        numeric  = (lp - lm) / (2 * eps)
        analytic = dW_head[j]
        denom    = max(abs(numeric), abs(analytic), 1e-8)
        rel_err  = abs(numeric - analytic) / denom
        max_rel_err = max(max_rel_err, rel_err)

    status = "PASSED" if max_rel_err < tol else "FAILED"
    print(f"[{cell_type.upper()}] max relative error: {max_rel_err:.6f}  -> {status}")
    if status == "FAILED":
        raise AssertionError(f"{cell_type} gradient check FAILED (err={max_rel_err:.6f})")
    return max_rel_err


if __name__ == "__main__":
    numerical_grad_check("lstm")
    numerical_grad_check("gru")
    print("All gradient checks passed.")
