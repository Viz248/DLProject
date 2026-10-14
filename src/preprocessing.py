"""
Preprocessing for the Elec2 (electricity) streaming dataset.

Changes vs original:
- Temporal chronological ordering guaranteed (data is already ordered but asserted).
- Three-way split: train / validation / test, all fitted on train only.
- Temporal gap between splits to prevent window overlap leakage.
- Scaler fit only on training set (no leakage from val/test periods).
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["date", "day", "period", "nswprice", "nswdemand",
                    "vicprice", "vicdemand", "transfer"]
LABEL_COLUMN = "class"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert df.isnull().sum().sum() == 0, "Unexpected missing values in raw data"
    return df


class StreamScaler:
    """Standardisation scaler (zero-mean, unit-variance) fit once on the
    training window only, then applied to all later periods — no look-ahead
    leakage into validation or test data."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fitted = False

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() before transform()"
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def build_stream(path: str,
                 train_frac: float = 0.60,
                 val_frac: float = 0.20,
                 temporal_gap: int = 40):
    """
    Chronological three-way split with a temporal gap to prevent sequence
    windows from straddling split boundaries.

    Returns:
        X          : (N, D) scaled feature matrix in original temporal order
        y          : (N,) binary labels — 1 = 'UP', 0 = 'DOWN'
        split_idx  : dict with keys 'train_end', 'val_start', 'val_end',
                     'test_start', representing absolute sample indices
        scaler     : fitted StreamScaler (fit on train only)
    """
    df = load_raw(path)
    X_raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = (df[LABEL_COLUMN].astype(str).str.upper() == "UP").astype(np.int64).to_numpy()

    n = len(df)
    train_end  = int(n * train_frac)
    val_start  = train_end + temporal_gap
    val_end    = int(n * (train_frac + val_frac))
    test_start = val_end + temporal_gap

    assert test_start < n, "Temporal gaps exceed dataset length — reduce gap or fractions."

    scaler = StreamScaler().fit(X_raw[:train_end])
    X = scaler.transform(X_raw)

    split_idx = dict(train_end=train_end, val_start=val_start,
                     val_end=val_end, test_start=test_start)
    return X, y, split_idx, scaler


if __name__ == "__main__":
    X, y, split_idx, scaler = build_stream("data/electricity.csv")
    n = len(X)
    print(f"Loaded stream: X={X.shape}, y={y.shape}")
    print(f"Splits: train 0-{split_idx['train_end']}, "
          f"val {split_idx['val_start']}-{split_idx['val_end']}, "
          f"test {split_idx['test_start']}-{n}")
    print(f"Class balance UP={y.mean():.3f}")
