"""
Label experiment: reference look-back of the degradation label.

    python -m src.label_experiment            # R in {20, 100, 500, 2000}, 2 seeds
    python -m src.label_experiment --quick    # smoke run

The original label compares the next `horizon` steps with the previous
`window` (=20) steps, so part of what it measures is mean reversion: a
"warn when the past error was low" score already has skill.  Here the
reference rate uses a longer look-back R (make_windows(reference_len=R)).

VALIDATION ONLY.  The test windows are built (the split needs them) but never
scored.  Early stopping uses the even validation windows; every number below
is computed on the odd validation windows.

Selection rule (fixed before running):
    skill_R = (AP_LSTM - AP_LowPastErr) / val positive rate
Pick the R with the largest mean skill_R; keep R = 20 unless the gain over
R = 20 is larger than the seed spread (max std of the two).  Then run the
full pipeline once with the chosen R.
"""
import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from src.preprocessing import build_online_stream
from src.mlp import MLP, run_stream
from src.sequence_dataset import make_windows, split_windows
from src.evaluate import save_json
from src.train import CFG, fit_norm, predict_all, train_rnn, window_summary_features


def _ap(y, s):
    return float(average_precision_score(y, s))


def main(refs=(20, 100, 500, 2000), seeds=2, quick=False):
    cfg = dict(CFG)
    if quick:
        cfg.update(epochs=1, train_stride=40)
        seeds = 1
    t_start = time.perf_counter()

    X, y, si, _ = build_online_stream(
        cfg["data"], warmup_frac=cfg["warmup_frac"], train_frac=cfg["train_frac"],
        val_frac=cfg["val_frac"], temporal_gap=cfg["temporal_gap"])
    ws = si["warmup_end"]
    mlp = MLP(n_in=X.shape[1], n_hidden=64, n_classes=2, lr=0.003,
              dropout=0.2, seed=0)
    mlp.fit_batch(X[:ws], y[:ws], epochs=30, batch_size=64)
    _, errors, confs, ents = run_stream(mlp, X, y, start=ws, online_update=True)

    out = {"selection_rule": "max mean (AP_LSTM - AP_LowPastErr)/val_pos_rate "
                             "on odd val windows; keep R=20 unless gain > seed std",
           "seeds": seeds, "epochs": cfg["epochs"], "variants": {}}
    for R in refs:
        Xw, yw, centers, pe, fe = make_windows(
            confs, ents, errors, window_len=cfg["window"], horizon=cfg["horizon"],
            degradation_delta=cfg["delta"], reference_len=R)
        (Xtr, ytr, _, _, _), (Xval, yval, _, pe_val, _), _ = split_windows(
            Xw, yw, centers + ws, pe, fe, val_start_abs=si["val_start"],
            test_start_abs=si["test_start"], train_end_abs=si["train_end"],
            val_end_abs=si["val_end"])
        ev, od = np.arange(0, len(Xval), 2), np.arange(1, len(Xval), 2)
        y_od = yval[od]
        prev = float(y_od.mean())

        pos_rate = float(ytr.mean())
        pos_weight = ((1.0 - pos_rate) / pos_rate) ** cfg["pos_weight_power"]
        ap_past = _ap(y_od, -pe_val[od])
        gbm = HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.05,
            class_weight="balanced", random_state=0
        ).fit(window_summary_features(Xtr), ytr)
        ap_gbm = _ap(y_od, gbm.predict_proba(window_summary_features(Xval[od]))[:, 1])

        mu, sd = fit_norm(Xtr)
        Ntr, Nval = (Xtr - mu) / sd, (Xval - mu) / sd
        tr_idx = np.arange(0, len(Ntr), cfg["train_stride"])
        aps = []
        for s in range(seeds):
            m, bep, _ = train_rnn("lstm", Ntr[tr_idx], ytr[tr_idx], Nval[ev],
                                  yval[ev], pos_weight, s, cfg)
            aps.append(_ap(y_od, predict_all(m, Nval[od])))
        aps = np.array(aps)
        skill = (aps - ap_past) / prev
        res = dict(reference_len=R, train_pos_rate=pos_rate, val_pos_rate=prev,
                   ap_lstm=float(aps.mean()), ap_lstm_std=float(aps.std()),
                   ap_low_past_err=ap_past, ap_gbm=ap_gbm,
                   lift_lstm=float(aps.mean() / prev),
                   lift_low_past_err=ap_past / prev, lift_gbm=ap_gbm / prev,
                   skill=float(skill.mean()), skill_std=float(skill.std()))
        out["variants"][str(R)] = res
        print(f"R={R:<5} pos tr/val={pos_rate:.3f}/{prev:.3f}  "
              f"AP LSTM={res['ap_lstm']:.3f}+-{res['ap_lstm_std']:.3f}  "
              f"LowPastErr={ap_past:.3f}  GBM={ap_gbm:.3f}  | lift LSTM="
              f"{res['lift_lstm']:.2f} past={res['lift_low_past_err']:.2f}  "
              f"skill={res['skill']:.3f}+-{res['skill_std']:.3f}  "
              f"[{time.perf_counter() - t_start:.0f}s]", flush=True)

    v = out["variants"]
    best = max(v, key=lambda k: v[k]["skill"])
    gain = v[best]["skill"] - v["20"]["skill"] if "20" in v else 0.0
    noise = max(v[best]["skill_std"], v.get("20", v[best])["skill_std"])
    chosen = best if gain > noise else "20"
    out.update(best_by_skill=int(best), gain_over_20=gain, noise=noise,
               chosen_reference_len=int(chosen))
    print(f"\nBest by validation skill: R={best} (gain over R=20: {gain:+.3f}, "
          f"seed spread {noise:.3f}) -> chosen R={chosen}")
    save_json(out, "label_experiment.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--refs", type=int, nargs="+", default=[20, 100, 500, 2000])
    a = ap.parse_args()
    main(tuple(a.refs), seeds=a.seeds, quick=a.quick)
