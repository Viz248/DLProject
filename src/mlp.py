"""
Base classifier: a small MLP (Dense -> ReLU -> Dropout -> Dense -> Softmax)
implemented from scratch in numpy so it can be updated online, one sample
(or mini-batch) at a time, as required by the streaming setup.

From its output probabilities we derive, for every timestep t:
    - prediction yhat_t
    - confidence C_t = max_k P(y=k | x_t)
    - entropy H_t   = -sum_k P(y=k|x_t) log P(y=k|x_t)
    - error E_t = 1[yhat_t != y_t]
"""
import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class MLP:
    def __init__(self, n_in: int, n_hidden: int = 32, n_classes: int = 2,
                 lr: float = 0.01, dropout: float = 0.2, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_classes))
        self.b2 = np.zeros(n_classes)
        self.lr = lr
        self.dropout = dropout
        self.rng = rng

    def _forward(self, x: np.ndarray, training: bool):
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        mask = np.ones_like(a1)
        if training and self.dropout > 0:
            mask = (self.rng.random(a1.shape) > self.dropout).astype(np.float64)
            a1 = a1 * mask / (1 - self.dropout)
        z2 = a1 @ self.W2 + self.b2
        p = softmax(z2)
        cache = (x, z1, a1, mask)
        return p, cache

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p, _ = self._forward(X, training=False)
        return p

    def partial_fit(self, x: np.ndarray, y: int):
        """One online SGD step on a single sample. Returns probs before update."""
        x = x.reshape(1, -1)
        p, (x_, z1, a1, mask) = self._forward(x, training=True)

        y_onehot = np.zeros_like(p)
        y_onehot[0, y] = 1.0
        dz2 = (p - y_onehot)  # softmax + CE gradient

        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        da1 = da1 * mask / (1 - self.dropout) if self.dropout > 0 else da1
        dz1 = da1 * (z1 > 0)

        dW1 = x_.T @ dz1
        db1 = dz1.sum(axis=0)

        clip = 5.0
        dW2 = np.clip(dW2, -clip, clip); db2 = np.clip(db2, -clip, clip)
        dW1 = np.clip(dW1, -clip, clip); db1 = np.clip(db1, -clip, clip)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return p[0]

    def fit_batch(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, batch_size: int = 32):
        """Offline warm-up training on the initial window."""
        n = len(X)
        for _ in range(epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                bidx = idx[start:start + batch_size]
                xb, yb = X[bidx], y[bidx]
                p, (x_, z1, a1, mask) = self._forward(xb, training=True)
                y_onehot = np.zeros_like(p)
                y_onehot[np.arange(len(yb)), yb] = 1.0
                dz2 = (p - y_onehot) / len(yb)

                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)
                da1 = dz2 @ self.W2.T
                da1 = da1 * mask / (1 - self.dropout) if self.dropout > 0 else da1
                dz1 = da1 * (z1 > 0)
                dW1 = x_.T @ dz1
                db1 = dz1.sum(axis=0)

                clip = 5.0
                dW2 = np.clip(dW2, -clip, clip); db2 = np.clip(db2, -clip, clip)
                dW1 = np.clip(dW1, -clip, clip); db1 = np.clip(db1, -clip, clip)

                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1


def confidence_entropy(p: np.ndarray):
    """p: (n_classes,) probability vector for one sample."""
    conf = float(p.max())
    eps = 1e-12
    ent = float(-np.sum(p * np.log(p + eps)))
    return conf, ent


def run_stream(mlp: MLP, X: np.ndarray, y: np.ndarray, start: int,
               online_update: bool = True):
    """
    Feed the stream through the MLP from index `start` onward (test-then-train
    protocol, the standard prequential evaluation for streaming data).

    Returns arrays: preds, errors, confidences, entropies
    """
    n = len(X) - start
    preds = np.zeros(n, dtype=np.int64)
    errors = np.zeros(n, dtype=np.int64)
    confs = np.zeros(n, dtype=np.float64)
    ents = np.zeros(n, dtype=np.float64)

    for i in range(n):
        t = start + i
        x_t, y_t = X[t], y[t]
        p = mlp.predict_proba(x_t.reshape(1, -1))[0]
        yhat = int(np.argmax(p))
        conf, ent = confidence_entropy(p)

        preds[i] = yhat
        errors[i] = int(yhat != y_t)
        confs[i] = conf
        ents[i] = ent

        if online_update:
            mlp.partial_fit(x_t, y_t)

    return preds, errors, confs, ents
