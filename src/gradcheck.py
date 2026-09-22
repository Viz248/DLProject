"""
Numerical gradient check for the custom LSTM and GRU cells (+head).
Run: python -m src.gradcheck
"""
import numpy as np
from src.rnn_layers import RNNDriftDetector


def numerical_grad_check(cell_type: str, T: int = 5, n_in: int = 2, n_hidden: int = 4,
                          eps: float = 1e-5, tol: float = 2e-2, seed: int = 0):
    rng = np.random.default_rng(seed)
    model = RNNDriftDetector(cell_type=cell_type, n_in=n_in, n_hidden=n_hidden, seed=seed)
    window = rng.normal(size=(T, n_in))
    label = 1

    h, cache = model.cell.forward(window)
    p = model.head.forward(h)
    dh, dW_head, db_head = model.head.backward(h, p, label)
    grads = model.cell.backward(dh, cache)

    def loss_fn():
        h_, _ = model.cell.forward(window)
        p_ = model.head.forward(h_)
        eps_ = 1e-12
        return -(label * np.log(p_ + eps_) + (1 - label) * np.log(1 - p_ + eps_))

    max_rel_err = 0.0
    for name, analytic_grad in grads.items():
        param = getattr(model.cell, name)
        it = np.nditer(param, flags=["multi_index"])
        # sample a subset of entries for speed
        n_checked = 0
        while not it.finished and n_checked < 6:
            idx = it.multi_index
            orig = param[idx]

            param[idx] = orig + eps
            loss_plus = loss_fn()
            param[idx] = orig - eps
            loss_minus = loss_fn()
            param[idx] = orig

            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = analytic_grad[idx]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            rel_err = abs(numeric - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)
            n_checked += 1
            it.iternext()

    print(f"[{cell_type.upper()}] max relative error over sampled params: {max_rel_err:.6f}")
    assert max_rel_err < tol, f"{cell_type} gradient check FAILED (err={max_rel_err})"
    print(f"[{cell_type.upper()}] gradient check PASSED")


if __name__ == "__main__":
    numerical_grad_check("lstm")
    numerical_grad_check("gru")
