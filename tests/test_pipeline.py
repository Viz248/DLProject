"""
Test suite for the DL concept-drift project.

Run from project root:
    python -m pytest tests/ -v
"""
import numpy as np
import pytest

# ------------------------------------------------------------------ #
#  Preprocessing                                                       #
# ------------------------------------------------------------------ #
class TestPreprocessing:
    def test_build_stream_shape(self):
        from src.preprocessing import build_stream
        X, y, split_idx, scaler = build_stream(
            "data/electricity.csv", train_frac=0.60,
            val_frac=0.20, temporal_gap=40)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == 45312

    def test_no_leakage_scaler_fitted_on_train_only(self):
        from src.preprocessing import build_stream
        import pandas as pd
        X, y, split_idx, scaler = build_stream(
            "data/electricity.csv", train_frac=0.60,
            val_frac=0.20, temporal_gap=40)
        train_end = split_idx["train_end"]
        # scaler mean/std should match training slice
        import pandas as pd
        df = pd.read_csv("data/electricity.csv")
        X_raw = df[["date","day","period","nswprice","nswdemand",
                     "vicprice","vicdemand","transfer"]].to_numpy(float)
        expected_mean = X_raw[:train_end].mean(axis=0)
        np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5)

    def test_labels_binary(self):
        from src.preprocessing import build_stream
        X, y, split_idx, scaler = build_stream("data/electricity.csv")
        assert set(np.unique(y)) <= {0, 1}

    def test_split_ordering(self):
        from src.preprocessing import build_stream
        X, y, split_idx, scaler = build_stream(
            "data/electricity.csv", temporal_gap=40)
        assert split_idx["train_end"]  < split_idx["val_start"]
        assert split_idx["val_start"]  < split_idx["val_end"]
        assert split_idx["val_end"]    < split_idx["test_start"]
        assert split_idx["test_start"] < len(X)

    def test_scaler_zero_std_handling(self):
        from src.preprocessing import StreamScaler
        X = np.ones((100, 3))   # constant columns
        s = StreamScaler().fit(X)
        assert np.all(s.std_ == 1.0)
        out = s.transform(X)
        assert np.all(out == 0.0)


# ------------------------------------------------------------------ #
#  MLP                                                                 #
# ------------------------------------------------------------------ #
class TestMLP:
    def _make_mlp(self, seed=0):
        from src.mlp import MLP
        return MLP(n_in=4, n_hidden=8, n_classes=2, lr=0.01, dropout=0.0, seed=seed)

    def test_predict_proba_shape(self):
        mlp = self._make_mlp()
        X   = np.random.default_rng(0).normal(size=(5, 4))
        p   = mlp.predict_proba(X)
        assert p.shape == (5, 2)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_probabilities_in_range(self):
        mlp = self._make_mlp()
        X   = np.random.default_rng(1).normal(size=(50, 4))
        p   = mlp.predict_proba(X)
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_partial_fit_returns_prob(self):
        mlp = self._make_mlp()
        x   = np.random.default_rng(2).normal(size=4)
        p   = mlp.partial_fit(x, 1)
        assert p.shape == (2,)
        assert abs(p.sum() - 1.0) < 1e-6

    def test_fit_batch_reduces_loss(self):
        from src.mlp import MLP
        rng = np.random.default_rng(42)
        X   = rng.normal(size=(200, 4))
        y   = rng.integers(0, 2, 200)
        mlp = MLP(n_in=4, n_hidden=16, lr=0.05, dropout=0.0, seed=42)
        def cross_entropy(m):
            p = m.predict_proba(X)
            return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        loss_before = cross_entropy(mlp)
        mlp.fit_batch(X, y, epochs=30, batch_size=32)
        loss_after  = cross_entropy(mlp)
        assert loss_after < loss_before

    def test_run_stream_lengths(self):
        from src.mlp import MLP, run_stream
        rng  = np.random.default_rng(0)
        X    = rng.normal(size=(100, 4))
        y    = rng.integers(0, 2, 100)
        mlp  = MLP(n_in=4, n_hidden=8, lr=0.01, dropout=0.0, seed=0)
        preds, errors, confs, ents = run_stream(mlp, X, y, start=20)
        assert len(preds) == len(errors) == len(confs) == len(ents) == 80

    def test_confidence_entropy(self):
        from src.mlp import confidence_entropy
        p = np.array([0.8, 0.2])
        c, e = confidence_entropy(p)
        assert abs(c - 0.8) < 1e-6
        assert e > 0


# ------------------------------------------------------------------ #
#  Sequence dataset                                                    #
# ------------------------------------------------------------------ #
class TestSequenceDataset:
    def _synthetic(self, n=500):
        rng  = np.random.default_rng(0)
        conf = rng.uniform(0.5, 1.0, n)
        ent  = rng.uniform(0.0, 0.7, n)
        err  = rng.binomial(1, 0.1, n)
        return conf, ent, err

    def test_output_shapes(self):
        from src.sequence_dataset import make_windows
        conf, ent, err = self._synthetic()
        X, y, centers, pe, fe = make_windows(conf, ent, err,
                                              window_len=20, horizon=20)
        assert X.ndim == 3
        assert X.shape[1] == 20
        assert X.shape[2] == 2
        assert len(y) == len(centers) == len(X)

    def test_no_future_leakage_in_input(self):
        """Each window input must only contain data strictly before center t."""
        from src.sequence_dataset import make_windows
        n    = 300
        conf = np.arange(n, dtype=float)   # identity so we can trace values
        ent  = np.zeros(n)
        err  = np.zeros(n, dtype=int)
        X, y, centers, pe, fe = make_windows(conf, ent, err,
                                              window_len=10, horizon=10)
        for i, t in enumerate(centers):
            # last conf value in window should be conf[t-1], not conf[t]
            assert X[i, -1, 0] == float(t - 1)

    def test_positive_label_logic(self):
        """Inject a degradation burst and confirm positive labels appear."""
        from src.sequence_dataset import make_windows
        n   = 600
        err = np.zeros(n, dtype=int)
        err[300:360] = 1   # high-error burst → future_err - past_err > 0.15
        conf = np.ones(n) * 0.9
        ent  = np.zeros(n)
        X, y, centers, pe, fe = make_windows(
            conf, ent, err, window_len=20, horizon=20, degradation_delta=0.15)
        assert y.sum() > 0, "Expected at least one positive label near degradation burst"

    def test_reference_len_default_is_original_label(self):
        from src.sequence_dataset import make_windows
        conf, ent, err = self._synthetic()
        a = make_windows(conf, ent, err, 20, 20, 0.15)
        b = make_windows(conf, ent, err, 20, 20, 0.15, reference_len=20)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)

    def test_reference_len_label(self):
        """Label = future rate minus mean(err[t-R:t]); inputs are unchanged."""
        from src.sequence_dataset import make_windows
        conf, ent, err = self._synthetic()
        R = 100
        X0, _, c0, _, _ = make_windows(conf, ent, err, 20, 20, 0.15)
        X, y, c, pe, fe = make_windows(conf, ent, err, 20, 20, 0.15,
                                       reference_len=R)
        np.testing.assert_array_equal(X, X0)
        np.testing.assert_array_equal(c, c0)
        for i, t in enumerate(c):
            ref = err[max(0, t - R):t].mean()
            assert pe[i] == pytest.approx(ref)
            assert fe[i] == pytest.approx(err[t:t + 20].mean())
            assert y[i] == int(fe[i] - ref > 0.15)

    def test_split_windows_no_overlap(self):
        from src.sequence_dataset import make_windows, split_windows
        conf, ent, err = self._synthetic()
        X, y, centers, pe, fe = make_windows(conf, ent, err)
        centers_abs = centers + 1000   # simulate absolute offsets
        (Xtr, ytr, ctr, _, _), \
        (Xval, yval, cval, _, _), \
        (Xte, yte, cte, _, _) = split_windows(
            X, y, centers_abs, pe, fe,
            val_start_abs=1200, test_start_abs=1350)
        # no overlap
        assert len(set(ctr) & set(cval)) == 0
        assert len(set(cval) & set(cte)) == 0
        assert len(set(ctr) & set(cte)) == 0


# ------------------------------------------------------------------ #
#  RNN layers and gradient check                                       #
# ------------------------------------------------------------------ #
class TestRNNLayers:
    def test_lstm_forward_shape(self):
        from src.rnn_layers import LSTMCell
        cell = LSTMCell(n_in=2, n_hidden=8, seed=0)
        X    = np.random.default_rng(0).normal(size=(10, 2))
        h, cache = cell.forward(X)
        assert h.shape == (8,)
        assert len(cache) == 10

    def test_gru_forward_shape(self):
        from src.rnn_layers import GRUCell
        cell = GRUCell(n_in=2, n_hidden=8, seed=0)
        X    = np.random.default_rng(0).normal(size=(10, 2))
        h, cache = cell.forward(X)
        assert h.shape == (8,)

    def test_predict_in_range(self):
        from src.rnn_layers import RNNDriftDetector
        for ct in ("lstm", "gru"):
            m   = RNNDriftDetector(cell_type=ct, n_in=2, n_hidden=8, seed=0)
            win = np.random.default_rng(0).normal(size=(20, 2))
            p   = m.predict(win)
            assert 0.0 <= p <= 1.0, f"{ct} probability out of range: {p}"

    def test_train_step_returns_finite_loss(self):
        from src.rnn_layers import RNNDriftDetector
        for ct in ("lstm", "gru"):
            m   = RNNDriftDetector(cell_type=ct, n_in=2, n_hidden=8,
                                   lr=0.01, seed=0)
            win = np.random.default_rng(0).normal(size=(20, 2))
            for label in (0, 1):
                loss = m.train_step(win, label)
                assert np.isfinite(loss), f"{ct} loss is not finite: {loss}"

    def test_gradient_check_lstm(self):
        from src.gradcheck import numerical_grad_check
        err = numerical_grad_check("lstm", T=5, n_in=2, n_hidden=4,
                                    eps=1e-5, tol=2e-2, n_sample=6)
        assert err < 0.02

    def test_gradient_check_gru(self):
        from src.gradcheck import numerical_grad_check
        err = numerical_grad_check("gru", T=5, n_in=2, n_hidden=4,
                                    eps=1e-5, tol=2e-2, n_sample=6)
        assert err < 0.02

    def test_get_set_params_roundtrip(self):
        from src.rnn_layers import RNNDriftDetector
        m     = RNNDriftDetector(cell_type="lstm", n_in=2, n_hidden=8, seed=0)
        params = m.get_params()
        # modify then restore
        m.head.W *= 2
        m.set_params(params)
        restored = m.get_params()
        for k in params:
            np.testing.assert_array_equal(params[k], restored[k])


# ------------------------------------------------------------------ #
#  Drift detectors                                                     #
# ------------------------------------------------------------------ #
class TestDriftDetectors:
    def test_adwin_detects_injected_drift(self):
        from src.drift_detectors import run_adwin, make_synthetic_stream
        errs, cp = make_synthetic_stream(seed=42)
        flags = run_adwin(errs)
        # must detect after the change point
        after_cp = np.where(flags)[0]
        assert len(after_cp) > 0
        assert after_cp[0] >= cp

    def test_ddm_detects_injected_drift(self):
        """DDM must raise at least one alarm on a stream with heavy drift.
        DDM can fire before the injected cp (during warm-up on the low-error
        segment) but must produce alarms overall."""
        from src.drift_detectors import run_ddm, make_synthetic_stream
        errs, cp = make_synthetic_stream(seed=42)
        flags, _ = run_ddm(errs)
        # At minimum, DDM must fire somewhere on the stream
        assert flags.sum() > 0
        # And at least one alarm must be at or after the change point
        assert np.any(np.where(flags)[0] >= cp)

    def test_no_change_stream_few_alarms(self):
        """On a stationary stream, false-alarm count should be low."""
        from src.drift_detectors import run_adwin, make_no_change_stream
        errs  = make_no_change_stream(n=5000, p=0.1, seed=0)
        flags = run_adwin(errs)
        # allow a handful of false alarms (<=5)
        assert flags.sum() <= 5

    def test_detector_metrics_perfect_detection(self):
        from src.drift_detectors import detector_metrics
        flags = np.zeros(1000, dtype=bool)
        flags[500] = True   # alarm exactly at change point
        m = detector_metrics(flags, [500], total_len=1000)
        assert m["n_true_detections"] == 1
        assert m["n_missed"] == 0
        assert m["n_false_alarms"] == 0
        assert m["mean_delay"] == 0

    def test_detector_metrics_missed(self):
        from src.drift_detectors import detector_metrics
        flags = np.zeros(1000, dtype=bool)
        m = detector_metrics(flags, [500], total_len=1000)
        assert m["n_missed"] == 1
        assert m["n_true_detections"] == 0
        assert m["mean_delay"] is None


# ------------------------------------------------------------------ #
#  Evaluate helpers                                                    #
# ------------------------------------------------------------------ #
class TestEvaluate:
    def test_select_threshold_returns_float(self):
        from src.evaluate import select_threshold
        rng   = np.random.default_rng(0)
        y     = rng.integers(0, 2, 200)
        probs = rng.uniform(0, 1, 200)
        thr   = select_threshold(y, probs, metric="f1")
        assert 0.0 <= thr <= 1.0

    def test_forecasting_metrics_all_zero(self):
        from src.evaluate import forecasting_metrics
        y    = np.array([0, 0, 0, 1, 1])
        prob = np.zeros(5)
        m    = forecasting_metrics(y, prob, threshold=0.5)
        assert m["recall_pos"] == 0.0

    def test_majority_class_metrics(self):
        from src.evaluate import majority_class_metrics
        y = np.array([0]*70 + [1]*30)
        m = majority_class_metrics(y)
        assert m["accuracy"] == pytest.approx(0.70)

    def test_mlp_metrics_perfect(self):
        from src.evaluate import mlp_metrics
        y    = np.array([0, 1, 0, 1])
        pred = np.array([0, 1, 0, 1])
        prob = np.array([0.1, 0.9, 0.1, 0.9])
        m    = mlp_metrics(y, pred, prob)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"]       == pytest.approx(1.0)


# ------------------------------------------------------------------ #
#  Fix-log regression tests (online split, causal scaling, no leakage) #
# ------------------------------------------------------------------ #
class TestOnlineStream:
    def test_split_ordering_and_warmup(self):
        from src.preprocessing import build_online_stream
        X, y, si, info = build_online_stream("data/electricity.csv")
        assert 0 < si["warmup_end"] < si["train_end"] < si["val_start"]
        assert si["val_start"] < si["val_end"] < si["test_start"] < si["n"]
        assert si["warmup_end"] <= 0.15 * si["n"]          # short warm-up
        assert X.shape == (45312, 8) and np.isfinite(X).all()

    def test_scaling_is_causal(self):
        """Changing FUTURE rows must not change already-scaled rows."""
        from src.preprocessing import causal_standardize
        rng = np.random.default_rng(0)
        A = rng.normal(size=(500, 3))
        B = A.copy()
        B[300:] += 50.0                       # corrupt the future
        ZA = causal_standardize(A, warmup_end=100)
        ZB = causal_standardize(B, warmup_end=100)
        np.testing.assert_allclose(ZA[:300], ZB[:300])

    def test_scaling_clipped(self):
        from src.preprocessing import causal_standardize
        A = np.ones((400, 2)); A[300:, 0] = 1e6
        Z = causal_standardize(A, warmup_end=100, clip=5.0)
        assert np.abs(Z).max() <= 5.0

    def test_training_window_set_is_large(self):
        """Regression: the old split produced ~20 training windows."""
        from src.preprocessing import build_online_stream
        from src.sequence_dataset import make_windows, split_windows
        X, y, si, _ = build_online_stream("data/electricity.csv")
        ws, n = si["warmup_end"], si["n"]
        rng = np.random.default_rng(0)
        m = n - ws
        conf, ent = rng.uniform(0.5, 1, m), rng.uniform(0, 0.7, m)
        err = rng.binomial(1, 0.13, m)
        Xw, yw, c, pe, fe = make_windows(conf, ent, err, 20, 20, 0.15)
        (Xtr, *_), (Xv, *_), (Xte, *_) = split_windows(
            Xw, yw, c + ws, pe, fe, si["val_start"], si["test_start"],
            si["train_end"], si["val_end"])
        assert len(Xtr) > 10_000
        assert len(Xv) > 3_000 and len(Xte) > 3_000
        assert len(Xtr) > 2 * len(Xv)          # train is the biggest split

    def test_window_splits_are_independent(self):
        """Label region of one split must not overlap the next split's input."""
        from src.sequence_dataset import make_windows, split_windows
        rng = np.random.default_rng(1)
        n = 3000
        X, y, c, pe, fe = make_windows(rng.uniform(.5, 1, n), rng.uniform(0, .7, n),
                                       rng.binomial(1, .1, n), 20, 20, 0.15)
        tr, va, te = split_windows(X, y, c, pe, fe, val_start_abs=1840,
                                   test_start_abs=2480, train_end_abs=1800,
                                   val_end_abs=2440)
        W = H = 20
        assert tr[2].max() + H <= va[2].min() - W
        assert va[2].max() + H <= te[2].min() - W


class TestEarlyWarning:
    def test_onsets_merge_runs(self):
        from src.early_warning import degradation_onsets
        centers = np.arange(100)
        labels = np.zeros(100, int)
        labels[10:15] = 1; labels[20:22] = 1; labels[70:75] = 1
        on = degradation_onsets(labels, centers, merge_gap=20)
        assert list(on) == [10, 70]            # 10-14 and 20-21 merged

    def test_alarm_refractory(self):
        from src.early_warning import alarms_from_flags
        flags = np.zeros(100, bool); flags[[10, 12, 15, 60]] = True
        assert list(alarms_from_flags(flags, 0, refractory=20)) == [10, 60]

    def test_evaluate_alarms_early_hit_and_false_alarm(self):
        from src.early_warning import evaluate_alarms
        r = evaluate_alarms(np.array([100, 500]), np.array([90, 300]),
                            span=1000, lead_max=30, tol=60)
        assert r["n_hits"] == 1 and r["early_rate"] == 0.5
        assert r["median_offset"] == -10 and r["false_alarms"] == 1

    def test_adaptive_alarms_are_causal(self):
        from src.early_warning import alarms_from_probs_adaptive
        rng = np.random.default_rng(0)
        p = rng.uniform(size=1500); c = np.arange(1500)
        q = p.copy(); q[1200:] = 1.0           # change the future only
        a = alarms_from_probs_adaptive(p, c, 0.95, 500, 100, 10)
        b = alarms_from_probs_adaptive(q, c, 0.95, 500, 100, 10)
        assert list(a[a < 1200]) == list(b[b < 1200])

    def test_chance_reference_in_unit_interval(self):
        from src.early_warning import chance_reference
        r = chance_reference(np.array([100, 400, 700]), 5, 0, 1000, 30, 60,
                             n_draws=50)
        assert 0.0 <= r["chance_event_recall"] <= 1.0


class TestBaselinesNoLeakage:
    def test_score_baseline_threshold_uses_validation_only(self):
        from src.evaluate import score_baseline_metrics
        rng = np.random.default_rng(0)
        s_val = rng.normal(size=500); y_val = (s_val > 1).astype(int)
        s_te = rng.normal(size=500);  y_a = (s_te > 1).astype(int)
        y_b = rng.integers(0, 2, 500)           # different test labels
        ra = score_baseline_metrics(s_val, y_val, s_te, y_a)
        rb = score_baseline_metrics(s_val, y_val, s_te, y_b)
        assert ra["threshold"] == rb["threshold"]

    def test_always_positive_and_majority(self):
        from src.evaluate import always_positive_metrics, majority_class_metrics
        y = np.array([0] * 80 + [1] * 20)
        assert always_positive_metrics(y)["precision_pos"] == pytest.approx(0.2)
        assert majority_class_metrics(y)["pr_auc"] == pytest.approx(0.2)

    def test_high_precision_threshold_in_range(self):
        from src.evaluate import select_threshold_precision
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 400); p = np.clip(y * 0.3 + rng.uniform(size=400) * .7, 0, 1)
        t = select_threshold_precision(y, p, min_recall=0.25)
        assert 0.0 <= t <= 1.0
