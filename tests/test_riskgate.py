"""Tests for the novelty algorithm RiskGate-CUSUM (src/riskgate.py)."""
import numpy as np
import pytest

from src.riskgate import RiskGateCUSUM, RiskGateParams, RiskRank, plain_cusum


def _step_stream(n=3000, change=1500, p0=0.10, p1=0.45, seed=0):
    rng = np.random.default_rng(seed)
    e = np.concatenate([rng.binomial(1, p0, change), rng.binomial(1, p1, n - change)]).astype(float)
    return e


class TestParams:
    def test_defaults_valid(self):
        RiskGateParams().validate()

    @pytest.mark.parametrize("bad", [dict(h0=0), dict(beta=1.0), dict(beta=-0.1),
                                     dict(gamma=-1), dict(u0=1.5), dict(ref_len=0),
                                     dict(min_ref=0), dict(min_ref=600)])
    def test_invalid_rejected(self, bad):
        with pytest.raises(AssertionError):
            RiskGateParams(**bad).validate()


class TestRiskRank:
    def test_zero_before_min_history(self):
        r = RiskRank(history=50, min_history=10)
        assert all(r.update(float(i)) == 0.0 for i in range(10))

    def test_range_and_monotone(self):
        rng = np.random.default_rng(0)
        r = RiskRank(history=500, min_history=50)
        for v in rng.uniform(size=300):
            r.update(float(v))
        lo, mid, hi = (RiskRank(500, 50) for _ in range(3))
        for rk in (lo, mid, hi):
            rk.warm(np.linspace(0, 1, 200))
        u_lo, u_mid, u_hi = lo.update(-1.0), mid.update(0.5), hi.update(2.0)
        assert 0.0 <= u_lo < u_mid < u_hi <= 1.0
        assert u_lo == 0.0 and u_hi == 1.0

    def test_causal_and_self_excluded(self):
        a, b = RiskRank(100, 5), RiskRank(100, 5)
        seq = list(np.random.default_rng(1).uniform(size=60))
        ua = [a.update(v) for v in seq]
        ub = [b.update(v) for v in seq[:40]] + [b.update(v + 100) for v in seq[40:]]
        assert ua[:40] == ub[:40]                      # future does not change the past

    def test_window_eviction_keeps_size(self):
        r = RiskRank(history=20, min_history=5)
        for v in np.random.default_rng(2).uniform(size=200):
            r.update(float(v))
        assert len(r._sorted) == 20 and len(r._fifo) == 20
        assert r._sorted == sorted(r._sorted)


class TestRiskGate:
    def test_reduces_to_plain_cusum(self):
        e = _step_stream()
        rng = np.random.default_rng(3)
        risks = rng.uniform(size=len(e))               # must be ignored
        p = RiskGateParams(h0=4.0, k=0.05, beta=0.0, gamma=0.0)
        a = RiskGateCUSUM(p).run(e, risks)
        b = plain_cusum(e, 4.0, 0.05)
        assert list(a) == list(b) and len(a) > 0

    def test_detects_step_change(self):
        e = _step_stream()
        al = plain_cusum(e, 4.0, 0.10)
        after = al[al >= 1500]
        assert len(after) > 0 and after[0] - 1500 < 120
        assert (al < 1500).sum() <= 2                 # few false alarms on the stationary part
        assert (al < 100).sum() == 0                  # no cold-start false alarm

    def test_online_equals_batch(self):
        e = _step_stream(1500, 700)
        r = np.random.default_rng(4).uniform(size=len(e))
        p = RiskGateParams(h0=3.0, beta=0.5, gamma=0.5)
        batch = RiskGateCUSUM(p).run(e, r)
        det = RiskGateCUSUM(p)
        online = [i for i in range(len(e)) if det.update(e[i], r[i])]
        assert list(batch) == online

    def test_strictly_causal(self):
        e = _step_stream(2000, 1000)
        r = np.random.default_rng(5).uniform(size=len(e))
        p = RiskGateParams(h0=3.0, beta=0.5, gamma=0.5)
        a = RiskGateCUSUM(p).run(e, r)
        e2, r2 = e.copy(), r.copy()
        e2[1500:] = 1.0 - e2[1500:]
        r2[1500:] = 1.0
        b = RiskGateCUSUM(p).run(e2, r2)
        assert list(a[a < 1500]) == list(b[b < 1500])

    def test_gating_never_delays_first_alarm(self):
        """With gamma=0, a higher rank gives a lower threshold, so the first
        alarm cannot come later than with rank 0 (same errors)."""
        e = _step_stream(3000, 1500, seed=7)
        p = RiskGateParams(h0=4.0, beta=0.8, gamma=0.0, rank_min_history=50)
        low = RiskGateCUSUM(p).run(e, np.zeros(len(e)))
        rng = np.random.default_rng(8)
        risky = np.sort(rng.uniform(size=len(e)))      # rising risk -> rank near 1
        high = RiskGateCUSUM(p).run(e, risky)
        assert len(low) > 0 and len(high) > 0
        assert high[0] <= low[0]

    def test_evidence_injection_alarms_without_errors(self):
        e = np.zeros(2000)
        risky = np.sort(np.random.default_rng(6).uniform(size=2000))
        on = RiskGateCUSUM(RiskGateParams(h0=2.0, k=0.0, beta=0.0, gamma=3.0,
                                          rank_min_history=50)).run(e, risky)
        off = RiskGateCUSUM(RiskGateParams(h0=2.0, k=0.0, beta=0.0, gamma=0.0,
                                           rank_min_history=50)).run(e, risky)
        assert len(on) > 0 and len(off) == 0

    def test_resets_and_refractory(self):
        e = np.concatenate([np.zeros(600), np.ones(400)])
        det = RiskGateCUSUM(RiskGateParams(h0=1.0, k=0.0, beta=0.0, gamma=0.0,
                                           ref_len=1000, refractory=30))
        al = [i for i in range(len(e)) if det.update(e[i], 0.0)]
        assert all(b - a >= 30 for a, b in zip(al, al[1:])) and len(al) >= 5
        assert det.S >= 0.0

    def test_initial_errors_set_reference(self):
        det = RiskGateCUSUM(RiskGateParams(ref_len=10, min_ref=5), initial_errors=[1.0] * 10)
        assert det.reference_rate() == pytest.approx(1.0)
        det.update(0.0, 0.0)
        assert det.reference_rate() == pytest.approx(0.9)

    def test_deterministic(self):
        e = _step_stream(1200, 600)
        r = np.random.default_rng(9).uniform(size=len(e))
        p = RiskGateParams(h0=3.0, beta=0.4, gamma=0.5)
        assert list(RiskGateCUSUM(p).run(e, r)) == list(RiskGateCUSUM(p).run(e, r))

    def test_length_mismatch_rejected(self):
        with pytest.raises(AssertionError):
            RiskGateCUSUM().run(np.zeros(5), np.zeros(4))


class TestTuning:
    def test_tune_family_returns_feasible_params(self):
        from src.novelty_experiment import tune_family, event_f1
        from src.early_warning import degradation_onsets
        e = _step_stream(4000, 2000, seed=11)
        centers = np.arange(1000, 3500)
        risks = np.random.default_rng(12).uniform(size=len(centers))
        onsets = np.array([2000])
        cfg = dict(refractory=20, lead_max=30, tol=60)
        grid = dict(h0=[3.0, 6.0], k=[0.05], beta=[0.0, 0.5], gamma=[0.0, 1.0])
        p, res = tune_family("CUSUM (plain)", e, centers, risks, onsets, cfg, budget=50.0, grid=grid)
        assert p.beta == 0.0 and p.gamma == 0.0
        assert res["alarms_per_1000"] <= 50.0 and 0.0 <= event_f1(res) <= 1.0
        p2, _ = tune_family("RiskGate-CUSUM", e, centers, risks, onsets, cfg, budget=50.0, grid=grid)
        assert p2.beta > 0.0 and p2.gamma > 0.0
