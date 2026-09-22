"""
From-scratch numpy implementations of an LSTM and a GRU, each followed by a
linear + sigmoid head that outputs a single "future degradation" probability
from a window of [confidence, entropy] pairs.

Both cells expose forward() and backward() and have been verified against
numerical gradient checks (see gradcheck.py).
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def dsigmoid(y):  # derivative given sigmoid output y
    return y * (1 - y)


def dtanh(y):  # derivative given tanh output y
    return 1 - y ** 2


class LSTMCell:
    """Standard LSTM: forget, input, candidate, output gates."""

    def __init__(self, n_in: int, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        z = n_in + n_hidden
        scale = np.sqrt(1.0 / z)
        self.Wf = rng.normal(0, scale, (z, n_hidden)); self.bf = np.zeros(n_hidden)
        self.Wi = rng.normal(0, scale, (z, n_hidden)); self.bi = np.zeros(n_hidden)
        self.Wc = rng.normal(0, scale, (z, n_hidden)); self.bc = np.zeros(n_hidden)
        self.Wo = rng.normal(0, scale, (z, n_hidden)); self.bo = np.zeros(n_hidden)
        self.n_in, self.n_hidden = n_in, n_hidden

    def params(self):
        return {"Wf": self.Wf, "bf": self.bf, "Wi": self.Wi, "bi": self.bi,
                "Wc": self.Wc, "bc": self.bc, "Wo": self.Wo, "bo": self.bo}

    def forward(self, X: np.ndarray):
        """X: (T, n_in) for a single sequence. Returns final hidden state h_T
        and a cache list for backprop-through-time."""
        T = X.shape[0]
        h = np.zeros(self.n_hidden)
        c = np.zeros(self.n_hidden)
        cache = []
        for t in range(T):
            z = np.concatenate([h, X[t]])
            f = sigmoid(z @ self.Wf + self.bf)
            i = sigmoid(z @ self.Wi + self.bi)
            g = np.tanh(z @ self.Wc + self.bc)
            o = sigmoid(z @ self.Wo + self.bo)
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            cache.append((z, f, i, g, o, c, c_new, h_new))
            h, c = h_new, c_new
        return h, cache

    def backward(self, dh_T: np.ndarray, cache):
        """BPTT given the gradient of the loss wrt the final hidden state only
        (our use case: one prediction per window, from h_T)."""
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        dh_next = dh_T.copy()
        dc_next = np.zeros(self.n_hidden)

        for t in reversed(range(len(cache))):
            z, f, i, g, o, c_prev, c_new, h_new = cache[t]
            tanh_c = np.tanh(c_new)

            do = dh_next * tanh_c
            dc = dh_next * o * dtanh(tanh_c) + dc_next

            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            do_raw = do * dsigmoid(o)
            df_raw = df * dsigmoid(f)
            di_raw = di * dsigmoid(i)
            dg_raw = dg * dtanh(g)

            grads["Wf"] += np.outer(z, df_raw); grads["bf"] += df_raw
            grads["Wi"] += np.outer(z, di_raw); grads["bi"] += di_raw
            grads["Wc"] += np.outer(z, dg_raw); grads["bc"] += dg_raw
            grads["Wo"] += np.outer(z, do_raw); grads["bo"] += do_raw

            dz = (df_raw @ self.Wf.T + di_raw @ self.Wi.T +
                  dg_raw @ self.Wc.T + do_raw @ self.Wo.T)
            dh_next = dz[:self.n_hidden]
            dc_next = dc_prev

        return grads


class GRUCell:
    """Standard GRU: reset, update, candidate gates (fewer params than LSTM,
    used as an ablation to test whether gains come from sequence modelling
    in general or LSTM's memory cell specifically)."""

    def __init__(self, n_in: int, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        z = n_in + n_hidden
        scale = np.sqrt(1.0 / z)
        self.Wr = rng.normal(0, scale, (z, n_hidden)); self.br = np.zeros(n_hidden)
        self.Wu = rng.normal(0, scale, (z, n_hidden)); self.bu = np.zeros(n_hidden)
        self.Wh = rng.normal(0, scale, (z, n_hidden)); self.bh = np.zeros(n_hidden)
        self.n_in, self.n_hidden = n_in, n_hidden

    def params(self):
        return {"Wr": self.Wr, "br": self.br, "Wu": self.Wu, "bu": self.bu,
                "Wh": self.Wh, "bh": self.bh}

    def forward(self, X: np.ndarray):
        T = X.shape[0]
        h = np.zeros(self.n_hidden)
        cache = []
        for t in range(T):
            z_in = np.concatenate([h, X[t]])
            r = sigmoid(z_in @ self.Wr + self.br)
            u = sigmoid(z_in @ self.Wu + self.bu)
            z_cand = np.concatenate([r * h, X[t]])
            h_tilde = np.tanh(z_cand @ self.Wh + self.bh)
            h_new = (1 - u) * h + u * h_tilde
            cache.append((z_in, z_cand, r, u, h_tilde, h))
            h = h_new
        return h, cache

    def backward(self, dh_T: np.ndarray, cache):
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        dh_next = dh_T.copy()

        for t in reversed(range(len(cache))):
            z_in, z_cand, r, u, h_tilde, h_prev = cache[t]

            dh_tilde = dh_next * u
            du = dh_next * (h_tilde - h_prev)
            dh_prev_direct = dh_next * (1 - u)

            dh_tilde_raw = dh_tilde * dtanh(h_tilde)
            grads["Wh"] += np.outer(z_cand, dh_tilde_raw); grads["bh"] += dh_tilde_raw

            dz_cand = dh_tilde_raw @ self.Wh.T
            dr_h = dz_cand[:self.n_hidden]  # d(r*h_prev)
            dr = dr_h * h_prev
            dh_prev_from_r = dr_h * r

            du_raw = du * dsigmoid(u)
            dr_raw = dr * dsigmoid(r)

            grads["Wu"] += np.outer(z_in, du_raw); grads["bu"] += du_raw
            grads["Wr"] += np.outer(z_in, dr_raw); grads["br"] += dr_raw

            dz_in = du_raw @ self.Wu.T + dr_raw @ self.Wr.T
            dh_prev_from_gates = dz_in[:self.n_hidden]

            dh_next = dh_prev_direct + dh_prev_from_r + dh_prev_from_gates

        return grads


class SequenceHead:
    """Linear + sigmoid readout on top of an LSTM/GRU final hidden state,
    producing P(future degradation)."""

    def __init__(self, n_hidden: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(1.0 / n_hidden), n_hidden)
        self.b = 0.0

    def forward(self, h: np.ndarray):
        z = h @ self.W + self.b
        return sigmoid(z)

    def backward(self, h: np.ndarray, y_pred: float, y_true: int):
        # binary cross-entropy + sigmoid gradient
        dz = (y_pred - y_true)
        dW = dz * h
        db = dz
        dh = dz * self.W
        return dh, dW, db


class RNNDriftDetector:
    """Wraps an LSTM or GRU cell + SequenceHead into a trainable sequence
    classifier: window of [C_t, H_t] pairs -> P(future degradation)."""

    def __init__(self, cell_type: str = "lstm", n_in: int = 2, n_hidden: int = 16,
                 lr: float = 0.05, seed: int = 0):
        assert cell_type in ("lstm", "gru")
        self.cell_type = cell_type
        self.cell = LSTMCell(n_in, n_hidden, seed) if cell_type == "lstm" \
            else GRUCell(n_in, n_hidden, seed)
        self.head = SequenceHead(n_hidden, seed + 1)
        self.lr = lr

    def predict(self, window: np.ndarray) -> float:
        h, _ = self.cell.forward(window)
        return self.head.forward(h)

    def train_step(self, window: np.ndarray, label: int) -> float:
        h, cache = self.cell.forward(window)
        p = self.head.forward(h)
        dh, dW_head, db_head = self.head.backward(h, p, label)
        grads = self.cell.backward(dh, cache)

        # gradient clipping (BPTT over up to 20 steps can otherwise blow up)
        clip = 1.0
        dW_head = np.clip(dW_head, -clip, clip)
        db_head = np.clip(db_head, -clip, clip)
        grads = {k: np.clip(g, -clip, clip) for k, g in grads.items()}

        # SGD update
        self.head.W -= self.lr * dW_head
        self.head.b -= self.lr * db_head
        for name, g in grads.items():
            setattr(self.cell, name, getattr(self.cell, name) - self.lr * g)

        eps = 1e-12
        loss = -(label * np.log(p + eps) + (1 - label) * np.log(1 - p + eps))
        return float(loss)
