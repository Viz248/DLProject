"""
Novelty experiment: RiskGate-CUSUM (see src/riskgate.py).

    python -m src.novelty_experiment              # full run (~6-8 min, 1 CPU core)
    python -m src.novelty_experiment --quick      # smoke run (~1 min)
    python -m src.novelty_experiment --seeds 1    # one forecaster seed

Protocol (no test leakage)
--------------------------
1. Same stream, MLP warm-up, causal scaling and window split as the main pipeline.
2. Train a GRU forecaster (per seed) on the training windows -> risk r_t for every
   validation / test window (strictly past inputs only).
3. For each detector family the parameters are tuned on the VALIDATION stream only:
       plain CUSUM        beta = gamma = 0
       gate-only          gamma = 0          (threshold gating)
       inject-only        beta = 0           (evidence injection)
       RiskGate-CUSUM     beta, gamma free   (the proposed method)
   Objective: event-level F1 (alarm precision vs event recall, same protocol as
   Table D of the report) subject to at most `budget` alarms per 1000 steps, so
   every variant gets the same alarm budget.
4. The tuned detectors are run ONCE on the test stream and compared with ADWIN,
   DDM, the forecaster alone and a naive "CUSUM OR forecaster" fusion.
"""
import argparse
import itertools
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.preprocessing import build_online_stream
from src.mlp import MLP, run_stream
from src.sequence_dataset import make_windows, split_windows
from src.drift_detectors import run_detectors_on_stream
from src.early_warning import (alarms_from_flags, alarms_from_probs_adaptive,
                               degradation_onsets, evaluate_alarms,
                               compare_methods, _apply_refractory)
from src.evaluate import save_json
from src.riskgate import RiskGateCUSUM, RiskGateParams
from src.train import CFG, fit_norm, predict_all, train_rnn

FIG_DIR = os.path.join("reports", "figures")

GRID = dict(h0=[2.0, 3.0, 4.0, 6.0, 8.0],
            k=[0.03, 0.06, 0.10],
            beta=[0.0, 0.3, 0.6, 0.85],
            gamma=[0.0, 0.5, 1.0, 2.0])

FAMILIES = {            # name -> predicate on (beta, gamma)
    "CUSUM (plain)":   lambda b, g: b == 0.0 and g == 0.0,
    "Gate-only":       lambda b, g: b > 0.0 and g == 0.0,
    "Inject-only":     lambda b, g: b == 0.0 and g > 0.0,
    "RiskGate-CUSUM":  lambda b, g: b > 0.0 and g > 0.0,
}


# ------------------------------------------------------------------ #
#  helpers                                                             #
# ------------------------------------------------------------------ #
def event_f1(res: Dict) -> float:
    p, r = res["alarm_precision"], res["event_recall"]
    if p != p or r != r or (p + r) == 0:          # NaN or zero
        return 0.0
    return float(2 * p * r / (p + r))


def run_gate(params: RiskGateParams, errors: np.ndarray, centers_rel: np.ndarray,
             risks: np.ndarray, warm_risks=None, trace: bool = False):
    """Run RiskGate-CUSUM over consecutive window centres.  The error at stream
    index t is e_t; the risk at centre t only used inputs before t."""
    start = int(centers_rel[0])
    det = RiskGateCUSUM(params,
                        initial_errors=errors[max(0, start - params.ref_len):start],
                        initial_risks=warm_risks)
    alarms, S, H, U = [], [], [], []
    for t, r in zip(centers_rel, risks):
        fired = det.update(errors[t], r)
        if fired:
            alarms.append(int(t))
        if trace:
            S.append(det.S); H.append(det.last_threshold); U.append(det.last_u)
    out = np.array(alarms, dtype=np.int64)
    return (out, np.array(S), np.array(H), np.array(U)) if trace else out


def tune_family(name: str, errors, centers_rel, risks, onsets, cfg, budget: float,
                grid=GRID) -> Tuple[RiskGateParams, Dict]:
    """Grid-search one family on the validation stream.  Returns best params and
    its validation result.  Feasible = alarms/1000 steps <= budget."""
    pred = FAMILIES[name]
    span = int(centers_rel[-1] - centers_rel[0] + 1)
    best, best_key = None, None
    for h0, k, beta, gamma in itertools.product(grid["h0"], grid["k"],
                                                grid["beta"], grid["gamma"]):
        if not pred(beta, gamma):
            continue
        p = RiskGateParams(h0=h0, k=k, beta=beta, gamma=gamma,
                           refractory=cfg["refractory"])
        al = run_gate(p, errors, centers_rel, risks)
        res = evaluate_alarms(onsets, al, span, cfg["lead_max"], cfg["tol"])
        rate = 1000.0 * len(al) / span
        feasible = rate <= budget
        # feasible configs first, then higher F1, then fewer alarms
        key = (feasible, event_f1(res) if feasible else -rate, -len(al))
        if best_key is None or key > best_key:
            best, best_key, best_res = p, key, dict(res, alarms_per_1000=rate,
                                                    event_f1=event_f1(res))
    return best, best_res


def _agg(per_seed: List[Dict]) -> Dict:
    out = {}
    for k in per_seed[0]:
        vals = [d[k] for d in per_seed]
        if all(isinstance(v, (int, float)) and v is not None for v in vals):
            a = np.array(vals, dtype=np.float64)
            out[k] = float(np.nanmean(a)) if not np.all(np.isnan(a)) else float("nan")
            out[k + "_std"] = float(np.nanstd(a)) if not np.all(np.isnan(a)) else float("nan")
        else:                                      # e.g. median_offset may be None
            nums = [v for v in vals if isinstance(v, (int, float))]
            out[k] = float(np.mean(nums)) if nums else None
    return out


# ------------------------------------------------------------------ #
def main(cfg_over: Optional[Dict] = None, seeds: int = 3, quick: bool = False,
         budget: float = 6.0):
    cfg = dict(CFG)
    cfg.update(cfg_over or {})
    grid = GRID
    if quick:
        cfg.update(epochs=1, train_stride=40)
        seeds = 1
        grid = dict(h0=[3.0, 6.0], k=[0.06], beta=[0.0, 0.6], gamma=[0.0, 1.0])
    os.makedirs("results", exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.perf_counter()
    banner = lambda s: print("\n" + "=" * 66 + f"\n{s}\n" + "=" * 66, flush=True)

    # ---------- stream, signals, windows (same as the main pipeline) ----------
    banner("1. Stream, MLP signals and windows")
    X, y, si, _ = build_online_stream(cfg["data"], warmup_frac=cfg["warmup_frac"],
                                      train_frac=cfg["train_frac"], val_frac=cfg["val_frac"],
                                      temporal_gap=cfg["temporal_gap"])
    ws = si["warmup_end"]
    mlp = MLP(n_in=X.shape[1], n_hidden=64, n_classes=2, lr=0.003, dropout=0.2, seed=0)
    mlp.fit_batch(X[:ws], y[:ws], epochs=30, batch_size=64)
    _, errors, confs, ents = run_stream(mlp, X, y, start=ws, online_update=True)
    Xw, yw, centers, pe, fe = make_windows(confs, ents, errors, cfg["window"], cfg["horizon"],
                                           cfg["delta"], reference_len=cfg.get("reference_len"))
    (Xtr, ytr, _, _, _), (Xval, yval, cval, pe_v, fe_v), (Xte, yte, cte, pe_t, fe_t) = split_windows(
        Xw, yw, centers + ws, pe, fe, si["val_start"], si["test_start"],
        si["train_end"], si["val_end"])
    cval_rel, cte_rel = cval - ws, cte - ws
    print(f"  windows train/val/test: {len(Xtr)}/{len(Xval)}/{len(Xte)}   MLP stream acc {1 - errors.mean():.4f}")

    ev = lambda f, p, c: degradation_onsets(((f - p) > cfg["event_delta"]).astype(int), c, cfg["merge_gap"])
    on_val, on_te = ev(fe_v, pe_v, cval_rel), ev(fe_t, pe_t, cte_rel)
    print(f"  severe-degradation events: val {len(on_val)}, test {len(on_te)}")

    # ---------- reactive detectors (seed independent) ----------
    det = run_detectors_on_stream(errors)
    from src.drift_detectors import run_adwin
    reactive = {
        "ADWIN": alarms_from_flags(det["adwin"]["flags"], 0, cfg["refractory"]),
        "ADWIN(d=.05)": alarms_from_flags(run_adwin(errors, delta=0.05), 0, cfg["refractory"]),
        "DDM": alarms_from_flags(det["ddm"]["flags"], 0, cfg["refractory"]),
    }

    mu, sd = fit_norm(Xtr)
    Ntr, Nval, Nte = (Xtr - mu) / sd, (Xval - mu) / sd, (Xte - mu) / sd
    tr_idx = np.arange(0, len(Ntr), cfg["train_stride"])
    v_sub = np.arange(0, len(Nval), 2)
    pos_rate = float(ytr.mean())
    pos_weight = ((1.0 - pos_rate) / pos_rate) ** cfg["pos_weight_power"]

    per_seed: Dict[str, List[Dict]] = {}
    tuned: Dict[str, List[Dict]] = {}
    trace_seed0 = None
    span_te = int(cte_rel[-1] - cte_rel[0] + 1)

    for s in range(seeds):
        banner(f"2. Forecaster (GRU) seed {s}: train -> risk series")
        t1 = time.perf_counter()
        model, bep, vap = train_rnn("gru", Ntr[tr_idx], ytr[tr_idx], Nval[v_sub], yval[v_sub],
                                    pos_weight, s, cfg)
        r_val, r_te = predict_all(model, Nval), predict_all(model, Nte)
        print(f"  best epoch {bep}, val AP {vap:.3f}  ({time.perf_counter() - t1:.0f}s)", flush=True)

        banner(f"3. Tune the four detector families on VALIDATION only (seed {s})")
        best: Dict[str, RiskGateParams] = {}
        for name in FAMILIES:
            p, res = tune_family(name, errors, cval_rel, r_val, on_val, cfg, budget, grid)
            best[name] = p
            print(f"  {name:<16} h0={p.h0} k={p.k} beta={p.beta} gamma={p.gamma} | val F1 {res['event_f1']:.3f} "
                  f"(P {res['alarm_precision']:.2f} R {res['event_recall']:.2f}, {res['alarms_per_1000']:.1f}/1000)", flush=True)

        banner(f"4. Evaluate ONCE on the test stream (seed {s})")
        alarm_sets = {}
        for name, p in best.items():
            alarm_sets[name] = run_gate(p, errors, cte_rel, r_te, warm_risks=r_val[-p.rank_history:])
            tuned.setdefault(name, []).append(p.to_dict())
        allp = np.concatenate([r_val, r_te]); allc = np.concatenate([cval_rel, cte_rel])
        fa = alarms_from_probs_adaptive(allp, allc, cfg["alarm_quantile"], cfg["alarm_history"], 300, cfg["refractory"])
        alarm_sets["Forecaster only"] = fa[fa >= cte_rel[0]]
        alarm_sets["Naive OR (CUSUM+Forecaster)"] = _apply_refractory(
            np.union1d(alarm_sets["CUSUM (plain)"], alarm_sets["Forecaster only"]), cfg["refractory"])
        alarm_sets.update(reactive)

        res = compare_methods(on_te, alarm_sets, start=int(cte_rel[0]), end=int(cte_rel[-1]) + 1,
                              lead_max=cfg["lead_max"], tol=cfg["tol"])
        for name, r in res.items():
            r["event_f1"] = event_f1(r)
            per_seed.setdefault(name, []).append(r)
        if s == 0:
            p = best["RiskGate-CUSUM"]
            _, S, H, U = run_gate(p, errors, cte_rel, r_te, warm_risks=r_val[-p.rank_history:], trace=True)
            trace_seed0 = dict(S=S, H=H, U=U, centers=cte_rel, onsets=on_te,
                               alarms=alarm_sets["RiskGate-CUSUM"], plain=alarm_sets["CUSUM (plain)"])
        for name in ("CUSUM (plain)", "RiskGate-CUSUM"):
            r = res[name]
            print(f"  {name:<16} alarms {r['n_alarms']:>3}  recall {r['event_recall']:.2f} (chance {r['chance_event_recall']:.2f}) "
                  f"alarm-prec {r['alarm_precision']:.2f}  F1 {r['event_f1']:.3f}", flush=True)

    # ---------- aggregate / report ----------
    banner(f"5. Test results (mean over {seeds} forecaster seed(s); {len(on_te)} events, {span_te} steps)")
    agg = {name: _agg(v) for name, v in per_seed.items()}
    print(f"{'Method':<28}{'Alarms':>7}{'Recall':>8}{'Chance':>8}{'AlPrec':>8}{'F1':>7}{'Early':>7}{'MedOff':>8}{'FA/1000':>9}")
    for name, m in agg.items():
        mo = m.get("median_offset")
        print(f"  {name:<26}{m['n_alarms']:>7.0f}{m['event_recall']:>8.2f}{m['chance_event_recall']:>8.2f}"
              f"{m['alarm_precision']:>8.2f}{m['event_f1']:>7.3f}{m['early_rate']:>7.2f}"
              f"{('n/a' if mo is None else f'{mo:.0f}'):>8}{m['false_alarms_per_1000']:>9.1f}")

    out = dict(config={k: cfg[k] for k in ("event_delta", "merge_gap", "lead_max", "tol", "refractory",
                                           "alarm_quantile", "alarm_history", "epochs", "train_stride")},
               budget_per_1000=budget, seeds=seeds, n_test_events=int(len(on_te)), n_val_events=int(len(on_val)),
               test_steps=span_te, grid=grid, tuned_params=tuned, methods=agg, per_seed=per_seed,
               runtime_s=float(time.perf_counter() - t0))
    save_json(out, "novelty_riskgate.json")
    if trace_seed0 is not None:
        plot_novelty(trace_seed0, agg)
    print(f"\nDone in {time.perf_counter() - t0:.0f}s -> results/novelty_riskgate.json, "
          f"{FIG_DIR}/novelty_riskgate.png")
    return out


def plot_novelty(tr: Dict, agg: Dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = tr["centers"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 10), gridspec_kw=dict(height_ratios=[1, 1.3, 1.3]))
    ax[0].plot(c, tr["U"], lw=0.7, color="tab:green")
    ax[0].axhline(0.8, color="gray", ls="--", lw=0.8)
    ax[0].set_ylabel("risk rank u_t"); ax[0].set_title("Forecaster risk (causal percentile rank)")
    ax[1].plot(c, tr["S"], lw=0.8, color="tab:blue", label="statistic S_t")
    ax[1].plot(c, tr["H"], lw=0.8, color="tab:red", label="gated threshold h_t")
    ax[1].set_ylabel("CUSUM"); ax[1].set_title("RiskGate-CUSUM internals on the test stream"); ax[1].legend(loc="upper right")
    for a in (ax[0], ax[1]):
        for o in tr["onsets"]:
            a.axvline(o, color="orange", alpha=0.35, lw=1)
        for al in tr["alarms"]:
            a.axvline(al, color="black", alpha=0.6, lw=0.8, ls=":")
    ax[1].set_xlabel("stream index (orange = degradation onsets, dotted = RiskGate alarms)")
    names = list(agg.keys()); x = np.arange(len(names))
    ax[2].bar(x - 0.27, [agg[n]["event_recall"] for n in names], 0.27, label="event recall")
    ax[2].bar(x, [agg[n]["alarm_precision"] if agg[n]["alarm_precision"] == agg[n]["alarm_precision"] else 0 for n in names], 0.27, label="alarm precision")
    ax[2].bar(x + 0.27, [agg[n]["event_f1"] for n in names], 0.27, label="event F1")
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax[2].legend(); ax[2].set_title("Test comparison (mean over forecaster seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "novelty_riskgate.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=float, default=6.0, help="max alarms per 1000 steps on validation")
    a = ap.parse_args()
    main(seeds=a.seeds, quick=a.quick, budget=a.budget)
