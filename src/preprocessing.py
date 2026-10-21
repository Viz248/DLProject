"""
Preprocessing for the Elec2 (electricity) streaming dataset.

Two entry points
----------------
build_stream()        legacy split: MLP warm-up on the first 60 % of the data.
build_online_stream() RECOMMENDED: short MLP warm-up (default 10 %) + causal
                      (past-only) scaling, so the monitoring signals exist for
                      ~90 % of the stream and the LSTM/GRU forecasters get a
                      real training set.  The legacy split left only ~20
                      training windows (signals only existed after the 60 %
                      mark), which made the forecasters effectively untrained.

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


def causal_standardize(X_raw: np.ndarray, warmup_end: int,
                       min_std: float = 1e-6,
                       clip: float = 5.0) -> np.ndarray:
    """
    Leak-free standardisation for a stream.

    * Samples inside the warm-up slice [0, warmup_end) are scaled with the
      warm-up statistics (this slice is the MLP's training data).
    * Every later sample t is scaled with the mean/std of X_raw[:t] only,
      i.e. strictly past data - exactly what a deployed system could compute.

    This avoids the blow-up of a scaler frozen on a short early slice
    (e.g. the monotone `date` column) while never using future information.
    Scaled values are clipped to +-`clip` standard deviations: the Victoria
    price/demand/transfer columns are constant for long stretches early on,
    so a regime change would otherwise produce 100+ sigma inputs.
    """
    n = len(X_raw)
    assert 2 <= warmup_end < n
    csum  = np.cumsum(X_raw, axis=0)
    csum2 = np.cumsum(X_raw ** 2, axis=0)
    k = np.arange(1, n + 1, dtype=np.float64)[:, None]
    mean = csum / k
    var  = np.maximum(csum2 / k - mean ** 2, 0.0)

    # stats "up to and including t-1" for t >= warmup_end
    mean_s = np.empty_like(mean)
    var_s  = np.empty_like(var)
    mean_s[:warmup_end] = mean[warmup_end - 1]
    var_s[:warmup_end]  = var[warmup_end - 1]
    mean_s[warmup_end:] = mean[warmup_end - 1:-1]
    var_s[warmup_end:]  = var[warmup_end - 1:-1]

    std_s = np.sqrt(var_s)
    std_s[std_s < min_std] = 1.0
    Z = (X_raw - mean_s) / std_s
    if clip is not None:
        Z = np.clip(Z, -clip, clip)
    return Z


def build_online_stream(path: str,
                        warmup_frac: float = 0.10,
                        train_frac: float = 0.60,
                        val_frac: float = 0.20,
                        temporal_gap: int = 40):
    """
    Streaming split used by the corrected pipeline.

        |--- warm-up ---|------------- monitored stream ----------------|
        0           warmup_end   [ train | gap | val | gap | test ]   n

    * The MLP is trained offline on [0, warmup_end) only.
    * The prequential pass starts at warmup_end, so confidence / entropy /
      error signals exist for the whole monitored stream.
    * The LSTM/GRU window datasets are split 60/20/20 *of the monitored
      stream* (train_frac / val_frac / remainder), with a gap between splits.

    Returns X (causally standardised), y, split_idx, info dict.
    """
    assert 0.0 < warmup_frac < 0.5
    df = load_raw(path)
    X_raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = (df[LABEL_COLUMN].astype(str).str.upper() == "UP").astype(np.int64).to_numpy()

    n = len(df)
    warmup_end = int(n * warmup_frac)
    m = n - warmup_end                       # monitored-stream length

    train_end  = warmup_end + int(m * train_frac)
    val_start  = train_end + temporal_gap
    val_end    = warmup_end + int(m * (train_frac + val_frac))
    test_start = val_end + temporal_gap
    assert test_start < n, "Temporal gaps exceed stream length."

    X = causal_standardize(X_raw, warmup_end)
    split_idx = dict(warmup_end=warmup_end, train_end=train_end,
                     val_start=val_start, val_end=val_end,
                     test_start=test_start, n=n)
    info = dict(scaler="causal_expanding", warmup_frac=warmup_frac)
    return X, y, split_idx, info


if __name__ == "__main__":
    X, y, split_idx, scaler = build_stream("data/electricity.csv")
    n = len(X)
    print(f"Loaded stream: X={X.shape}, y={y.shape}")
    print(f"Splits: train 0-{split_idx['train_end']}, "
          f"val {split_idx['val_start']}-{split_idx['val_end']}, "
          f"test {split_idx['test_start']}-{n}")
    print(f"Class balance UP={y.mean():.3f}")
