"""
From-scratch numpy LSTM and GRU cells with a sigmoid readout head.
Verified against numerical gradient checks (see gradcheck.py).

Architecture
------------
Input window : (T, n_in)  — e.g. [C_t, H_t] pairs
RNN cell      : processes the sequence step-by-step, yields final h_T
SequenceHead  : linear + sigmoid on h_T -> P(future degradation in [0,1])
"""
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def dsigmoid(y: np.ndarray) -> np.ndarray:
    return y * (1.0 - y)


def dtanh(y: np.ndarray) -> np.ndarray:
    return 1.0 - y ** 2


# ====================================================================== #
#  LSTM cell                                                               #
# ====================================================================== #
class LSTMCell:
    """Vanilla LSTM with forget / input / candidate / output gates."""

    def __init__(self, n_in: int, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        z = n_in + n_hidden
        scale = np.sqrt(1.0 / z)
        self.Wf = rng.normal(0, scale, (z, n_hidden)); self.bf = np.ones(n_hidden) * 1.0
        self.Wi = rng.normal(0, scale, (z, n_hidden)); self.bi = np.zeros(n_hidden)
        self.Wc = rng.normal(0, scale, (z, n_hidden)); self.bc = np.zeros(n_hidden)
        self.Wo = rng.normal(0, scale, (z, n_hidden)); self.bo = np.zeros(n_hidden)
        self.n_in = n_in; self.n_hidden = n_hidden

    def params(self):
        return {"Wf": self.Wf, "bf": self.bf, "Wi": self.Wi, "bi": self.bi,
                "Wc": self.Wc, "bc": self.bc, "Wo": self.Wo, "bo": self.bo}

    def forward(self, X: np.ndarray):
        """X: (T, n_in). Returns (h_T, cache)."""
        T = X.shape[0]
        h = np.zeros(self.n_hidden)
        c = np.zeros(self.n_hidden)
        cache = []
        for t in range(T):
            z  = np.concatenate([h, X[t]])
            f  = sigmoid(z @ self.Wf + self.bf)
            i  = sigmoid(z @ self.Wi + self.bi)
            g  = np.tanh(z @ self.Wc + self.bc)
            o  = sigmoid(z @ self.Wo + self.bo)
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            cache.append((z, f, i, g, o, c, c_new, h_new))
            h, c = h_new, c_new
        return h, cache

    def backward(self, dh_T: np.ndarray, cache):
        """BPTT from gradient of loss w.r.t. final hidden state h_T."""
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        dh = dh_T.copy()
        dc = np.zeros(self.n_hidden)

        for t in reversed(range(len(cache))):
            z, f, i, g, o, c_prev, c_new, h_new = cache[t]
            tanh_c = np.tanh(c_new)

            do = dh * tanh_c
            dc = dh * o * dtanh(tanh_c) + dc   # carry dc forward (backward in time)

            df    = dc * c_prev
            di    = dc * g
            dg    = dc * i
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
            dh = dz[:self.n_hidden]
            dc = dc_prev

        return grads


# ====================================================================== #
#  GRU cell                                                                #
# ====================================================================== #
class GRUCell:
    """Standard GRU (reset, update, candidate gates)."""

    def __init__(self, n_in: int, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        z = n_in + n_hidden
        scale = np.sqrt(1.0 / z)
        self.Wr = rng.normal(0, scale, (z, n_hidden)); self.br = np.zeros(n_hidden)
        self.Wu = rng.normal(0, scale, (z, n_hidden)); self.bu = np.zeros(n_hidden)
        self.Wh = rng.normal(0, scale, (z, n_hidden)); self.bh = np.zeros(n_hidden)
        self.n_in = n_in; self.n_hidden = n_hidden

    def params(self):
        return {"Wr": self.Wr, "br": self.br, "Wu": self.Wu, "bu": self.bu,
                "Wh": self.Wh, "bh": self.bh}

    def forward(self, X: np.ndarray):
        T = X.shape[0]
        h = np.zeros(self.n_hidden)
        cache = []
        for t in range(T):
            z_in    = np.concatenate([h, X[t]])
            r       = sigmoid(z_in @ self.Wr + self.br)
            u       = sigmoid(z_in @ self.Wu + self.bu)
            z_cand  = np.concatenate([r * h, X[t]])
            h_tilde = np.tanh(z_cand @ self.Wh + self.bh)
            h_new   = (1 - u) * h + u * h_tilde
            cache.append((z_in, z_cand, r, u, h_tilde, h))
            h = h_new
        return h, cache

    def backward(self, dh_T: np.ndarray, cache):
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        dh = dh_T.copy()

        for t in reversed(range(len(cache))):
            z_in, z_cand, r, u, h_tilde, h_prev = cache[t]

            dh_tilde         = dh * u
            du               = dh * (h_tilde - h_prev)
            dh_prev_direct   = dh * (1 - u)

            dh_tilde_raw = dh_tilde * dtanh(h_tilde)
            grads["Wh"] += np.outer(z_cand, dh_tilde_raw)
            grads["bh"] += dh_tilde_raw

            dz_cand = dh_tilde_raw @ self.Wh.T
            dr_h    = dz_cand[:self.n_hidden]
            dr      = dr_h * h_prev
            dh_prev_from_r = dr_h * r

            du_raw = du * dsigmoid(u)
            dr_raw = dr * dsigmoid(r)

            grads["Wu"] += np.outer(z_in, du_raw); grads["bu"] += du_raw
            grads["Wr"] += np.outer(z_in, dr_raw); grads["br"] += dr_raw

            dz_in = du_raw @ self.Wu.T + dr_raw @ self.Wr.T
            dh = dh_prev_direct + dh_prev_from_r + dz_in[:self.n_hidden]

        return grads


# ====================================================================== #
#  Readout head                                                            #
# ====================================================================== #
class SequenceHead:
    """Linear + sigmoid readout: h_T -> P(future degradation)."""

    def __init__(self, n_hidden: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(1.0 / n_hidden), n_hidden)
        self.b = 0.0

    def forward(self, h: np.ndarray) -> float:
        return float(sigmoid(float(h @ self.W + self.b)))

    def backward(self, h: np.ndarray, y_pred: float, y_true: int):
        dz = y_pred - y_true
        return dz * self.W, dz * h, float(dz)   # dh, dW, db


# ====================================================================== #
#  Full forecaster                                                          #
# ====================================================================== #
class RNNDriftDetector:
    """Wraps a cell + head; optionally uses class weighting to handle
    imbalanced degradation-label distributions."""

    def __init__(self, cell_type: str = "lstm", n_in: int = 2,
                 n_hidden: int = 16, lr: float = 0.01, seed: int = 0,
                 pos_weight: float = 1.0):
        assert cell_type in ("lstm", "gru")
        self.cell_type = cell_type
        self.cell = (LSTMCell(n_in, n_hidden, seed)
                     if cell_type == "lstm"
                     else GRUCell(n_in, n_hidden, seed))
        self.head = SequenceHead(n_hidden, seed + 1)
        self.lr = lr
        self.pos_weight = pos_weight   # multiply loss / grad for positive class

    def predict(self, window: np.ndarray) -> float:
        h, _ = self.cell.forward(window)
        return self.head.forward(h)

    def train_step(self, window: np.ndarray, label: int) -> float:
        h, cache = self.cell.forward(window)
        p = self.head.forward(h)
        dh, dW_head, db_head = self.head.backward(h, p, label)

        # class-weighted gradient scaling
        if label == 1 and self.pos_weight != 1.0:
            dh       *= self.pos_weight
            dW_head  *= self.pos_weight
            db_head  *= self.pos_weight

        grads = self.cell.backward(dh, cache)

        clip = 1.0
        np.clip(dW_head, -clip, clip, out=dW_head)
        db_head = float(np.clip(db_head, -clip, clip))
        grads = {k: np.clip(g, -clip, clip) for k, g in grads.items()}

        self.head.W -= self.lr * dW_head
        self.head.b -= self.lr * db_head
        for name, g in grads.items():
            setattr(self.cell, name, getattr(self.cell, name) - self.lr * g)

        eps = 1e-12
        loss = -(label * np.log(p + eps) + (1 - label) * np.log(1 - p + eps))
        if label == 1:
            loss *= self.pos_weight
        return float(loss)

    def get_params(self) -> dict:
        d = {k: v.copy() for k, v in self.cell.params().items()}
        d["head_W"] = self.head.W.copy()
        d["head_b"] = float(self.head.b)
        return d

    def set_params(self, d: dict):
        for k in self.cell.params():
            setattr(self.cell, k, d[k].copy())
        self.head.W = d["head_W"].copy()
        self.head.b = float(d["head_b"])
