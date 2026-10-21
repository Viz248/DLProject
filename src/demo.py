"""
Drift demo: inject drift of a known type and size into the Elec2 test stream,
run the full monitoring system on it, and give a verdict.

    python -m src.demo                              # all scenarios -> table + demo/demo_data.js
    python -m src.demo --scenario sudden --severity 0.25    # one scenario, detailed

Needs a finished `python -m src.train` run (uses artifacts/lstm_params.npz) and
docs/novelty/novelty_riskgate.json (RiskGate / CUSUM settings tuned on validation
by src.novelty_experiment).  The first run trains the seed-0 GRU forecaster that
RiskGate was tuned with (~1.5 min) and caches it in artifacts/.

What runs, for every scenario
-----------------------------
1. The MLP is warmed up and run prequentially exactly as in the pipeline up to
   the start of the test stream (same seeds, so the same model state).
2. The test stream is modified from ONSET onward (or left alone for "none"),
   and the MLP keeps running test-then-train on it, adapting as it goes.
3. Detectors on the MLP's error stream:
   - existing: ADWIN and DDM (fresh detectors at the start of the test stream)
     and a plain one-sided CUSUM;
   - ours: RiskGate-CUSUM (src/riskgate.py), the CUSUM gated by the GRU
     forecaster's risk, and the LSTM forecaster on its own (alarm when its
     output exceeds the 95th percentile of its own past 2000 outputs).
   CUSUM and RiskGate use the settings tuned on validation in
   src/novelty_experiment.py, started with the errors and risks from before
   the test stream, exactly as in that experiment.
4. verdict() compares the REF steps before the onset with the MON steps after.
   Only ADWIN and DDM confirm drift there: they test for a sustained change,
   while CUSUM/RiskGate were tuned to catch short degradation episodes, which
   also occur naturally in the undrifted stream.

Scenarios (ground truth is known because we inject it)
------------------------------------------------------
none     the real test stream, untouched
sudden   concept drift: at ONSET the up/down rule reverses during a block of
         half-hour slots covering `severity` of the day (labels inverted
         there; every slot is equally common, so this is `severity` of samples)
gradual  the same, but the affected share of the day grows over RAMP steps
covshift the price features move by `severity` standard deviations; labels
         are untouched (input drift that may or may not hurt the model)
"""
import argparse
import copy
import json
import os

import numpy as np

from src.preprocessing import build_online_stream, load_raw
from src.mlp import MLP, run_stream
from src.drift_detectors import run_adwin, run_ddm
from src.sequence_dataset import make_windows
from src.early_warning import alarms_from_probs_adaptive
from src.rnn_layers import RNNDriftDetector
from src.riskgate import RiskGateParams
from src.novelty_experiment import run_gate
from src.train import CFG, fit_norm, predict_all, train_rnn

ONSET, REF, MON, RAMP = 4000, 2000, 2000, 1500   # test-relative steps
MIN_EFFECT = 0.05      # a drift must raise the error rate by >= 5 percentage points
METHODS = ["ADWIN", "DDM", "CUSUM", "RiskGate", "LSTM"]
NOVELTY_JSON = os.path.join("docs", "novelty", "novelty_riskgate.json")
PRICES = [3, 5]   # feature columns nswprice, vicprice

SCENARIOS = {
    "none":     dict(title="No drift (real stream)", severities=[0.0]),
    "sudden":   dict(title="Sudden concept drift", severities=[0.10, 0.25, 0.50]),
    "gradual":  dict(title="Gradual concept drift", severities=[0.10, 0.25, 0.50]),
    "covshift": dict(title="Feature shift (prices)", severities=[1.0, 2.5]),
}


def describe(kind, sev):
    if kind == "none":
        return "Nothing injected: the real Elec2 test stream."
    if kind == "sudden":
        return (f"From step {ONSET} the up/down price rule reverses during "
                f"{sev:.0%} of the day's half-hour slots.")
    if kind == "gradual":
        return (f"From step {ONSET} the rule starts reversing; the affected share "
                f"of the day grows to {sev:.0%} over {RAMP} steps.")
    return (f"From step {ONSET} both price features shift by +{sev:g} standard "
            f"deviations; labels are unchanged.")


# ------------------------------------------------------------------ #
#  Drift injection                                                     #
# ------------------------------------------------------------------ #
def inject(Xt, yt, period, kind, sev):
    """Return modified copies of the test-stream features and labels, and the
    mask of samples whose label was flipped.  `period` is the raw Elec2
    time-of-day column (48 half-hour slots scaled to 0..1)."""
    X, y = Xt.copy(), yt.copy()
    flip = np.zeros(len(y), dtype=bool)
    if kind in ("sudden", "gradual"):
        slot = np.rint(period * 47).astype(int)          # 0..47
        i = np.arange(len(X)) - ONSET
        frac = np.where(i < 0, 0.0,
                        sev if kind == "sudden" else sev * np.minimum(1.0, (i + 1) / RAMP))
        flip = slot < np.rint(frac * 48)
        y[flip] = 1 - y[flip]
    elif kind == "covshift":
        X[ONSET:, PRICES] += sev
    return X, y, flip


# ------------------------------------------------------------------ #
#  Verdict                                                             #
# ------------------------------------------------------------------ #
def block_bootstrap_ci(a, b, block=50, reps=2000, seed=0):
    """95% CI of mean(b) - mean(a) with a moving-block bootstrap (errors are
    autocorrelated, so resampling single steps would make the CI too narrow)."""
    rng = np.random.default_rng(seed)

    def means(x):
        nb = len(x) // block
        starts = rng.integers(0, len(x) - block + 1, size=(reps, nb))
        idx = starts[:, :, None] + np.arange(block)
        return x[idx].mean(axis=(1, 2))

    d = means(b.astype(float)) - means(a.astype(float))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def first_after(times, t0):
    after = [int(t) for t in times if t >= t0]
    return (after[0] - t0) if after else None


def verdict(err_ref, err_mon, alarms):
    """
    alarms: {"ADWIN": array, "DDM": array} of alarm times (test-relative).

    DRIFT      error rate significantly higher (95% CI above 0), by at least
               MIN_EFFECT, AND at least one established detector (ADWIN/DDM)
               raised an alarm after the onset.
    NO DRIFT   every signal agrees nothing changed: no significant rise, the
               estimated rise is below MIN_EFFECT, and neither detector fired.
    UNCERTAIN  the signals disagree (e.g. a significant rise that the detectors
               did not confirm, or a detector alarm with only a small change).

    Errors arrive in bursts, so the 95% CI on a 2000-step window is about
    +-4 pp wide: "no drift" means no evidence of drift, not proof of none.
    """
    p0, p1 = float(err_ref.mean()), float(err_mon.mean())
    d = p1 - p0
    lo, hi = block_bootstrap_ci(err_ref, err_mon)
    fired = {k: bool(np.any((v >= ONSET) & (v < ONSET + MON))) for k, v in alarms.items()}
    any_fired = any(fired.values())
    significant = lo > 0
    big = d >= MIN_EFFECT

    if significant and big and any_fired:
        label = "DRIFT"
        sev = "severe" if d >= 0.20 else "moderate" if d >= 0.10 else "mild"
        reason = (f"Error rate rose significantly by {d * 100:.1f} percentage points "
                  f"and {' and '.join(k for k, f in fired.items() if f)} confirmed it.")
    elif not significant and not big and not any_fired:
        label, sev = "NO DRIFT", None
        reason = (f"Error rate changed by {d * 100:+.1f} pp, which is not significant "
                  f"(95% CI {lo * 100:+.1f} to {hi * 100:+.1f} pp), and neither ADWIN "
                  f"nor DDM fired.")
    else:
        label, sev = "UNCERTAIN", None
        if significant and big:
            reason = (f"Error rate rose by {d * 100:.1f} pp, but neither ADWIN nor DDM "
                      f"confirmed a change.")
        elif any_fired and not big:
            reason = (f"{' and '.join(k for k, f in fired.items() if f)} fired, but "
                      f"the error rate only changed by {d * 100:+.1f} pp.")
        elif significant:
            reason = (f"Error rate rose by {d * 100:.1f} pp: statistically real, but "
                      f"below the {MIN_EFFECT * 100:.0f} pp needed to call it drift.")
        else:
            reason = (f"Error rate rose by {d * 100:.1f} pp, but the rise is not "
                      f"statistically significant (95% CI {lo * 100:+.1f} to {hi * 100:+.1f} pp).")
    return dict(label=label, severity=sev, reason=reason,
                err_before=p0, err_after=p1, change_pp=d * 100,
                change_rel=(d / p0) if p0 > 0 else None,
                acc_before=1 - p0, acc_after=1 - p1,
                acc_change_rel=((1 - p1) - (1 - p0)) / (1 - p0),
                ci_pp=[lo * 100, hi * 100], significant=significant,
                detectors_fired=fired)


# ------------------------------------------------------------------ #
#  Running the system                                                  #
# ------------------------------------------------------------------ #
class System:
    """Builds the MLP state at the start of the test stream and the trained
    forecasters once, then runs any number of scenarios from that state."""

    def __init__(self, artifacts="artifacts"):
        cfg = CFG
        X, y, si, _ = build_online_stream(
            cfg["data"], warmup_frac=cfg["warmup_frac"], train_frac=cfg["train_frac"],
            val_frac=cfg["val_frac"], temporal_gap=cfg["temporal_gap"])
        ws, ts = si["warmup_end"], si["test_start"]
        mlp = MLP(n_in=X.shape[1], n_hidden=64, n_classes=2, lr=0.003,
                  dropout=0.2, seed=0)
        mlp.fit_batch(X[:ws], y[:ws], epochs=30, batch_size=64)
        _, err, conf, ent = run_stream(mlp, X[:ts], y[:ts], start=ws, online_update=True)
        self.mlp_at_test = mlp
        self.Xt, self.yt = X[ts:], y[ts:]
        self.period = load_raw(cfg["data"])["period"].to_numpy(float)[ts:]
        # input normalisation of the forecasters = statistics of the training windows
        Xw, yw, c, _, _ = make_windows(conf, ent, err, cfg["window"], cfg["horizon"],
                                       cfg["delta"])
        tr = c + ws < si["train_end"]
        va = (c + ws >= si["val_start"]) & (c + ws < si["val_end"])
        self.mu, self.sd = fit_norm(Xw[tr])
        Ntr, ytr = (Xw[tr] - self.mu) / self.sd, yw[tr]
        Nval, yval = (Xw[va] - self.mu) / self.sd, yw[va]
        # 20 steps of history so the first test window is complete
        self.hist = (conf[-cfg["window"]:], ent[-cfg["window"]:], err[-cfg["window"]:])
        self.prefix_err = err
        self.lstm = RNNDriftDetector("lstm", n_in=2, n_hidden=cfg["n_hidden"])
        self.lstm.set_params(dict(np.load(os.path.join(artifacts, "lstm_params.npz"))))

        # GRU forecaster that RiskGate was tuned with: seed 0, as in novelty_experiment
        gpath = os.path.join(artifacts, "gru_seed0_params.npz")
        pos_rate = float(ytr.mean())
        pos_weight = ((1.0 - pos_rate) / pos_rate) ** cfg["pos_weight_power"]
        self.gru = RNNDriftDetector("gru", n_in=2, n_hidden=cfg["n_hidden"],
                                    lr=cfg["lr"], seed=0, pos_weight=pos_weight)
        if os.path.exists(gpath):
            self.gru.set_params(dict(np.load(gpath)))
        else:
            print("Training the seed-0 GRU forecaster for RiskGate (first run only)...",
                  flush=True)
            tr_idx = np.arange(0, len(Ntr), cfg["train_stride"])
            v_sub = np.arange(0, len(Nval), 2)
            self.gru, _, _ = train_rnn("gru", Ntr[tr_idx], ytr[tr_idx], Nval[v_sub],
                                       yval[v_sub], pos_weight, 0, cfg)
            np.savez(gpath, **self.gru.get_params())
        self.val_risk = predict_all(self.gru, Nval)

        with open(NOVELTY_JSON) as f:
            tuned = json.load(f)["tuned_params"]
        self.p_rg = RiskGateParams(**tuned["RiskGate-CUSUM"][0])
        self.p_cusum = RiskGateParams(**tuned["CUSUM (plain)"][0])
        self.cfg = cfg

    def run(self, kind, sev):
        cfg = self.cfg
        X, y, flipped = inject(self.Xt, self.yt, self.period, kind, sev)
        mlp = copy.deepcopy(self.mlp_at_test)
        _, err, conf, ent = run_stream(mlp, X, y, start=0, online_update=True)

        alarms = {"ADWIN": np.flatnonzero(run_adwin(err)),
                  "DDM": np.flatnonzero(run_ddm(err)[0])}

        w = cfg["window"]
        c_all = np.concatenate([self.hist[0], conf])
        e_all = np.concatenate([self.hist[1], ent])
        r_all = np.concatenate([self.hist[2], err])
        Xw, _, cen, _, _ = make_windows(c_all, e_all, r_all, w, cfg["horizon"], cfg["delta"])
        cen = cen - w                                   # test-relative prediction time
        Nw = (Xw - self.mu) / self.sd
        probs = predict_all(self.lstm, Nw)
        lstm_alarms = alarms_from_probs_adaptive(
            probs, cen, cfg["alarm_quantile"], cfg["alarm_history"], 300, cfg["refractory"])

        # CUSUM and RiskGate continue from the pre-test errors/risks, as in the experiment
        g_risk = predict_all(self.gru, Nw)
        off = len(self.prefix_err)
        errs_all = np.concatenate([self.prefix_err, err])
        warm = self.val_risk[-self.p_rg.rank_history:]
        found = {**alarms,
                 "CUSUM": run_gate(self.p_cusum, errs_all, cen + off, g_risk, warm) - off,
                 "RiskGate": run_gate(self.p_rg, errs_all, cen + off, g_risk, warm) - off,
                 "LSTM": lstm_alarms}

        ref, mon = slice(ONSET - REF, ONSET), slice(ONSET, ONSET + MON)
        v = verdict(err[ref], err[mon], alarms)
        in_ref = lambda a: int(np.sum((a >= ONSET - REF) & (a < ONSET)))
        in_mon = lambda a: int(np.sum((a >= ONSET) & (a < ONSET + MON)))
        roll = np.convolve(err, np.ones(200) / 200, mode="valid")   # 200-step error rate
        mon_roll = roll[ONSET:ONSET + MON - 200]
        pk = int(np.argmax(mon_roll))
        v.update(
            delay={k: first_after(found[k], ONSET) for k in METHODS},
            alarms_before={k: in_ref(found[k]) for k in METHODS},
            alarms_after={k: in_mon(found[k]) for k in METHODS},
            peak_err=float(mon_roll[pk]), peak_at=pk,
            flipped_share=float(flipped[ONSET:ONSET + MON].mean()),
            conf_before=float(conf[ref].mean()), conf_after=float(conf[mon].mean()),
            risk_before=float(probs[(cen >= ONSET - REF) & (cen < ONSET)].mean()),
            risk_after=float(probs[(cen >= ONSET) & (cen < ONSET + MON)].mean()),
        )

        # downsampled series for the dashboard: steps ONSET-REF .. ONSET+MON
        lo_, hi_ = ONSET - REF, ONSET + MON
        step = 10
        ts = np.arange(lo_, hi_, step)
        rerr = np.convolve(err, np.ones(100) / 100, mode="full")[:len(err)]  # trailing 100
        rconf = np.convolve(conf, np.ones(100) / 100, mode="full")[:len(conf)]
        pr = np.full(len(err), np.nan)
        pr[cen] = probs
        rrisk = np.array([np.nanmean(pr[max(0, t - 50):t + 1]) for t in ts])
        win = lambda a: [int(t) for t in a if lo_ <= t < hi_]
        series = dict(t=ts.tolist(), err=np.round(rerr[ts], 4).tolist(),
                      conf=np.round(rconf[ts], 4).tolist(),
                      risk=np.round(rrisk, 4).tolist(),
                      alarms={k: win(found[k]) for k in METHODS})
        return v, series


def _fmt_delay(d):
    return "no alarm" if d is None else f"+{d} steps"


def print_detail(kind, sev, v):
    title = SCENARIOS[kind]["title"] + ("" if kind == "none" else f"  (severity {sev:g})")
    print("\n" + "=" * 70 + f"\n{title}\n  {describe(kind, sev)}\n" + "-" * 70)
    tag = v["label"] + (f" ({v['severity']})" if v["severity"] else "")
    if kind in ("sudden", "gradual"):
        print(f"  Ground truth: {v['flipped_share']:.1%} of labels flipped in the "
              f"{MON} steps after onset")
    print(f"  VERDICT: {tag}\n  {v['reason']}")
    print(f"  Accuracy     {v['acc_before']:.1%} -> {v['acc_after']:.1%}   "
          f"({v['acc_change_rel']:+.1%} relative)")
    rel = "" if v["change_rel"] is None else f", {v['change_rel']:+.0%} relative"
    print(f"  Error rate   {v['err_before']:.1%} -> {v['err_after']:.1%}   "
          f"({v['change_pp']:+.1f} pp{rel}; 95% CI {v['ci_pp'][0]:+.1f} to {v['ci_pp'][1]:+.1f} pp)")
    print(f"  Worst 200-step error rate after onset: {v['peak_err']:.1%} at +{v['peak_at']} steps")
    print("  First alarm after onset: " + ", ".join(
        f"{k} {_fmt_delay(v['delay'][k])}" for k in METHODS if k != "LSTM"))
    b, a = v["alarms_before"], v["alarms_after"]
    print(f"  Alarms {REF} steps before / {MON} after onset: " + ", ".join(
        f"{k} {b[k]}/{a[k]}" for k in METHODS))
    print(f"  Label-free signals: confidence {v['conf_before']:.3f} -> {v['conf_after']:.3f}, "
          f"LSTM risk {v['risk_before']:.3f} -> {v['risk_after']:.3f}")


def main(only=None, sev=None, out="demo/demo_data.js"):
    print("Preparing the monitoring system (MLP warm-up + prequential pass)...", flush=True)
    sysm = System()
    runs = []
    for kind, spec in SCENARIOS.items():
        if only and kind != only:
            continue
        for s in ([sev] if (only and sev is not None) else spec["severities"]):
            v, series = sysm.run(kind, s)
            print_detail(kind, s, v)
            runs.append(dict(id=f"{kind}-{s:g}", kind=kind, severity=s,
                             title=spec["title"], description=describe(kind, s),
                             verdict=v, series=series))

    print("\n" + "=" * 70 + "\nSUMMARY\n" + "-" * 70)
    print(f"  {'Scenario':<28}{'Verdict':<22}{'Accuracy':>20}{'Change':>10}")
    for r in runs:
        v = r["verdict"]
        name = r["title"] if r["kind"] == "none" else f"{r['title']} {r['severity']:g}"
        tag = v["label"] + (f" ({v['severity']})" if v["severity"] else "")
        print(f"  {name:<28}{tag:<22}{v['acc_before']:>9.1%} -> {v['acc_after']:<7.1%}"
              f"{v['change_pp']:>+8.1f}pp")

    if only is None and out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        meta = dict(onset=ONSET, ref=REF, mon=MON, ramp=RAMP, min_effect_pp=MIN_EFFECT * 100)
        with open(out, "w", encoding="utf-8") as f:
            f.write("window.DEMO = " + json.dumps(dict(meta=meta, runs=runs)) + ";\n")
        print(f"\nDashboard data -> {out}  (open demo/index.html)")
    return runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS))
    ap.add_argument("--severity", type=float)
    a = ap.parse_args()
    main(a.scenario, a.severity)
