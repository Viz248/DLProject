"""
Base classifier: MLP (Dense->ReLU->Dropout->Dense->Softmax) in numpy.
Prequential evaluation (test-then-train) over the streaming portion.

Monitoring signals produced per timestep t:
    C_t  = confidence = max_k P(y=k | x_t)
    H_t  = prediction entropy = -sum_k P(y=k | x_t) log P(y=k | x_t)
    E_t  = error indicator = 1[yhat_t != y_t]
"""
import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class MLP:
    def __init__(self, n_in: int, n_hidden: int = 32, n_classes: int = 2,
                 lr: float = 0.003, dropout: float = 0.2, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in),       (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden),   (n_hidden, n_classes))
        self.b2 = np.zeros(n_classes)
        self.lr = lr
        self.dropout = dropout
        self.rng = rng

    # ------------------------------------------------------------------ #
    def _forward(self, x: np.ndarray, training: bool):
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        mask = np.ones_like(a1)
        if training and self.dropout > 0:
            keep = 1.0 - self.dropout
            mask = (self.rng.random(a1.shape) < keep).astype(np.float64) / keep
            a1 = a1 * mask
        z2 = a1 @ self.W2 + self.b2
        p = softmax(z2)
        return p, (x, z1, a1, mask)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p, _ = self._forward(X, training=False)
        return p

    # ------------------------------------------------------------------ #
    def _backward_and_update(self, p, cache, y_oh, batch_size):
        x_, z1, a1, mask = cache
        dz2 = (p - y_oh) / batch_size

        dW2 = a1.T @ dz2;  db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T * mask
        dz1 = da1 * (z1 > 0)
        dW1 = x_.T @ dz1;  db1 = dz1.sum(0)

        clip = 5.0
        for g in (dW1, db1, dW2, db2):
            np.clip(g, -clip, clip, out=g)

        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    def partial_fit(self, x: np.ndarray, y: int):
        x = x.reshape(1, -1)
        p, cache = self._forward(x, training=True)
        y_oh = np.zeros_like(p);  y_oh[0, y] = 1.0
        self._backward_and_update(p, cache, y_oh, batch_size=1)
        return p[0]

    def fit_batch(self, X: np.ndarray, y: np.ndarray,
                  epochs: int = 20, batch_size: int = 32):
        n = len(X)
        for ep in range(epochs):
            idx = self.rng.permutation(n)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                bidx = idx[start:start + batch_size]
                xb, yb = X[bidx], y[bidx]
                p, cache = self._forward(xb, training=True)
                y_oh = np.zeros_like(p)
                y_oh[np.arange(len(yb)), yb] = 1.0
                eps = 1e-12
                total_loss += -np.sum(y_oh * np.log(p + eps))
                self._backward_and_update(p, cache, y_oh, batch_size=len(yb))


def confidence_entropy(p: np.ndarray):
    """p: (n_classes,) probability vector."""
    conf = float(p.max())
    eps = 1e-12
    ent = float(-np.sum(p * np.log(p + eps)))
    return conf, ent


def run_stream(mlp: MLP, X: np.ndarray, y: np.ndarray, start: int,
               online_update: bool = True):
    """
    Prequential evaluation (test-then-train) from index `start`.
    Returns: preds, errors, confidences, entropies (all relative to `start`).
    """
    n = len(X) - start
    preds  = np.zeros(n, dtype=np.int64)
    errors = np.zeros(n, dtype=np.int64)
    confs  = np.zeros(n, dtype=np.float64)
    ents   = np.zeros(n, dtype=np.float64)

    for i in range(n):
        t = start + i
        x_t, y_t = X[t], y[t]
        p = mlp.predict_proba(x_t.reshape(1, -1))[0]
        yhat = int(np.argmax(p))
        conf, ent = confidence_entropy(p)

        preds[i]  = yhat
        errors[i] = int(yhat != y_t)
        confs[i]  = conf
        ents[i]   = ent

        if online_update:
            mlp.partial_fit(x_t, y_t)

    return preds, errors, confs, ents
