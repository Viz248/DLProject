"""
Preprocessing for the Elec2 (electricity) streaming dataset.

The raw CSV is already numeric and has no missing values, but this module
still performs the full pipeline expected in the report: temporal ordering,
feature scaling (fit on an initial "batch" window only, to respect the
streaming setting), and label encoding.
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
    """Min-max style scaler fit once on an initial warm-up window, then
    applied to the rest of the stream (no leakage from future samples)."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def build_stream(path: str, warmup_frac: float = 0.1):
    """
    Returns:
        X: (N, D) scaled feature matrix, in original temporal order
        y: (N,) binary labels, 1 = 'UP', 0 = 'DOWN'
        warmup_idx: number of samples used to fit the MLP initially
    """
    df = load_raw(path)
    X_raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = (df[LABEL_COLUMN].astype(str).str.upper() == "UP").astype(np.int64).to_numpy()

    n = len(df)
    warmup_idx = int(n * warmup_frac)

    scaler = StreamScaler().fit(X_raw[:warmup_idx])
    X = scaler.transform(X_raw)

    return X, y, warmup_idx, scaler


if __name__ == "__main__":
    X, y, warmup_idx, scaler = build_stream("data/electricity.csv")
    print(f"Loaded stream: X={X.shape}, y={y.shape}, warmup={warmup_idx}")
    print(f"Class balance: UP={y.mean():.3f}")
