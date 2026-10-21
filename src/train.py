"""
End-to-end pipeline - run from the project root:
    python -m src.train                # full run
    python -m src.train --quick        # ~1 minute smoke run

Execution order
---------------
1.  Load Elec2; short MLP warm-up slice + causal scaling; monitored stream
    split 60/20/20 with temporal gaps (no leakage)
2.  Warm-up train the MLP on the warm-up slice only
3.  Prequential (test-then-train) pass over the whole monitored stream ->
    confidence, entropy and error signals
4.  ADWIN and DDM on the error stream
5.  Sliding-window datasets (inputs: past conf/entropy, label: future
    degradation) split by window centre
6.  Class weight; baselines (majority, always-warn, past-error, rolling
    entropy / confidence, gradient-boosting reference)
7.  Train LSTM and GRU (several seeds, class-weighted loss, early stopping on
    validation PR-AUC, threshold tuned on validation F1)
8.  Monitoring-signal ablation
9.  Early-warning analysis: forecaster alarms vs ADWIN/DDM alarms
10. Synthetic drift benchmark (known change points)
11. Save artifacts, plots, result tables

Why this differs from the first version: the old pipeline warmed the MLP up on
60 % of the stream, so monitoring signals only existed for the last 40 % and
the "training" window set had ~20 windows.  See README "Fix log".
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score

from src.preprocessing import build_online_stream
from src.mlp import MLP, run_stream
from src.drift_detectors import (run_detectors_on_stream, run_adwin,
                                 make_synthetic_stream, make_no_change_stream,
                                 detector_metrics)
from src.sequence_dataset import make_windows, split_windows
from src.rnn_layers import RNNDriftDetector
from src.early_warning import (alarms_from_probs, alarms_from_flags,
                               degradation_onsets, compare_methods,
                               alarms_from_probs_adaptive)
from src.evaluate import (
    mlp_metrics, forecasting_metrics, majority_class_metrics,
    always_positive_metrics, score_baseline_metrics, select_threshold,
    select_threshold_precision,
    save_json, plot_mlp_over_time, plot_signals_and_detectors,
    plot_confusion_matrix, plot_pr_curve, plot_degradation_probs,
    plot_synthetic_drift_experiment, plot_model_comparison, plot_ablation,
    plot_early_warning, print_table_a, print_table_b, print_table_c,
    print_table_d,
)

ARTIFACTS_DIR = "artifacts"

# ------------------------------ configuration ------------------------------ #
CFG = dict(
    data="data/electricity.csv",
    warmup_frac=0.10,      # MLP warm-up slice (fraction of the whole dataset)
    train_frac=0.60, val_frac=0.20, temporal_gap=40,
    window=20, horizon=20, delta=0.15,          # degradation label (see README)
    reference_len=None,    # None = original label (past `window` steps); see src/label_experiment.py
    n_hidden=32, lr=0.002, lr_decay=0.75,
    epochs=8, train_stride=3, seeds=3,
    pos_weight_power=0.5,  # pos_weight = ((1-p)/p) ** power  (1.0 = full balance)
    hp_min_recall=0.25,    # high-precision row: best val precision with recall >= this
    event_delta=0.30, merge_gap=40,              # 'severe degradation' events
    lead_max=30, tol=60, refractory=20,          # early-warning scoring
    alarm_quantile=0.95, alarm_history=2000,     # adaptive alarm rule (see early_warning)
)


def _ensure():
    for d in (ARTIFACTS_DIR, "results", os.path.join("reports", "figures")):
        os.makedirs(d, exist_ok=True)


# ------------------------------ helpers ------------------------------------ #
def fit_norm(Xtr):
    flat = Xtr.reshape(-1, Xtr.shape[2])
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def predict_all(model, X):
    return np.array([model.predict(w) for w in X])


def train_rnn(cell_type, Xtr, ytr, Xv_sub, yv_sub, pos_weight, seed, cfg):
    """Per-sample SGD with lr decay; keeps the epoch with the best validation
    PR-AUC (early stopping on validation only)."""
    model = RNNDriftDetector(cell_type=cell_type, n_in=Xtr.shape[2],
                             n_hidden=cfg["n_hidden"], lr=cfg["lr"],
                             seed=seed, pos_weight=pos_weight)
    rng = np.random.default_rng(seed)
    best_ap, best_params, best_ep = -1.0, model.get_params(), -1
    for ep in range(cfg["epochs"]):
        model.lr = cfg["lr"] * (cfg["lr_decay"] ** ep)
        for i in rng.permutation(len(Xtr)):
            model.train_step(Xtr[i], int(ytr[i]))
        ap = average_precision_score(yv_sub, predict_all(model, Xv_sub))
        if ap > best_ap:
            best_ap, best_params, best_ep = ap, model.get_params(), ep
    model.set_params(best_params)
    return model, best_ep, float(best_ap)


def window_summary_features(Xw):
    """Hand-crafted summary of a window - only for the non-neural reference."""
    c, h = Xw[:, :, 0], Xw[:, :, 1]
    q = max(Xw.shape[1] // 4, 2)
    return np.column_stack([c.mean(1), c.std(1), c.min(1), c[:, -q:].mean(1),
                            c[:, -q:].mean(1) - c[:, :q].mean(1),
                            h.mean(1), h.max(1), h[:, -q:].mean(1),
                            h[:, -q:].mean(1) - h[:, :q].mean(1)])


# ================================================================== #
def main(cfg=None, quick=False, resume=False):
    cfg = dict(CFG if cfg is None else cfg)
    if quick:
        cfg.update(epochs=1, train_stride=40, seeds=1)
    _ensure()
    t_start = time.perf_counter()

    banner = lambda s: print("\n" + "=" * 66 + f"\n{s}\n" + "=" * 66)

    # ---------------------------------------------------------------- #
    banner("STEP 1 - Load Elec2, warm-up slice + causal scaling")
    X, y, si, info = build_online_stream(
        cfg["data"], warmup_frac=cfg["warmup_frac"],
        train_frac=cfg["train_frac"], val_frac=cfg["val_frac"],
        temporal_gap=cfg["temporal_gap"])
    n = len(X)
    ws, train_end = si["warmup_end"], si["train_end"]
    val_start, val_end, test_start = si["val_start"], si["val_end"], si["test_start"]
    print(f"  Total samples   : {n}")
    print(f"  MLP warm-up     : 0-{ws}  ({ws} samples, scaler = causal expanding)")
    print(f"  Window train    : {ws}-{train_end}")
    print(f"  Window val      : {val_start}-{val_end}")
    print(f"  Window test     : {test_start}-{n}")
    print(f"  Label balance   : UP={y.mean():.3f}  DOWN={1 - y.mean():.3f}")

    # ---------------------------------------------------------------- #
    banner("STEP 2 - MLP warm-up on the warm-up slice only")
    mlp = MLP(n_in=X.shape[1], n_hidden=64, n_classes=2, lr=0.003,
              dropout=0.2, seed=0)
    mlp.fit_batch(X[:ws], y[:ws], epochs=30, batch_size=64)
    print("  Warm-up training complete.")

    # ---------------------------------------------------------------- #
    banner("STEP 3 - Prequential evaluation over the monitored stream")
    preds, errors, confs, ents = run_stream(mlp, X, y, start=ws,
                                            online_update=True)
    print(f"  Monitored stream length   : {len(errors)}")
    print(f"  Overall streaming accuracy: {1 - errors.mean():.4f}")
    test_rel = test_start - ws
    y_ts, preds_ts, confs_ts = y[test_start:], preds[test_rel:], confs[test_rel:]
    prob1_ts = np.where(preds_ts == 1, confs_ts, 1.0 - confs_ts)
    mlp_m = mlp_metrics(y_ts, preds_ts, prob1_ts)
    mlp_m["streaming_accuracy_all"] = float(1 - errors.mean())
    print_table_a(mlp_m)
    save_json(mlp_m, "table_a_mlp_metrics.json")
    plot_mlp_over_time(errors, confs, ents)

    # ---------------------------------------------------------------- #
    banner("STEP 4 - ADWIN and DDM on the error stream")
    det_out = run_detectors_on_stream(errors)
    adwin_flags, ddm_flags = det_out["adwin"]["flags"], det_out["ddm"]["flags"]
    adwin_sens = run_adwin(errors, delta=0.05)
    print(f"  ADWIN alarms: {adwin_flags.sum()}   DDM alarms: {ddm_flags.sum()}   "
          f"ADWIN(delta=0.05): {adwin_sens.sum()}")

    # ---------------------------------------------------------------- #
    banner("STEP 5 - Sliding-window datasets")
    Xw, yw, centers, pe, fe = make_windows(
        confs, ents, errors, window_len=cfg["window"], horizon=cfg["horizon"],
        degradation_delta=cfg["delta"], reference_len=cfg.get("reference_len"))
    centers_abs = centers + ws
    (Xtr, ytr, ctr, pe_tr, _), (Xval, yval, cval, pe_val, _), \
    (Xte, yte, cte, pe_te, fe_te) = split_windows(
        Xw, yw, centers_abs, pe, fe, val_start_abs=val_start,
        test_start_abs=test_start, train_end_abs=train_end, val_end_abs=val_end)
    print(f"  All windows   : {len(Xw)}  pos={yw.mean():.3f}")
    print(f"  Train windows : {len(Xtr)}  pos={ytr.mean():.3f}")
    print(f"  Val windows   : {len(Xval)}  pos={yval.mean():.3f}")
    print(f"  Test windows  : {len(Xte)}  pos={yte.mean():.3f}")
    assert len(Xtr) > 1000, "Training window set unexpectedly small"

    # ---------------------------------------------------------------- #
    banner("STEP 6 - Baselines and class weight")
    pos_rate = float(ytr.mean())
    pos_weight = ((1.0 - pos_rate) / pos_rate) ** cfg["pos_weight_power"]
    print(f"  Train positive rate: {pos_rate:.3f}  pos_weight={pos_weight:.2f}")

    fcast_table = {}
    fcast_table["Majority"]    = majority_class_metrics(yte)
    fcast_table["AlwaysWarn"]  = always_positive_metrics(yte)
    # non-learned scores computed from window contents (strictly past data);
    # "PastErr" needs ground-truth labels, the others do not.
    fcast_table["LowPastErr*"] = score_baseline_metrics(-pe_val, yval, -pe_te, yte)
    fcast_table["RollEnt"]     = score_baseline_metrics(
        Xval[:, :, 1].mean(1), yval, Xte[:, :, 1].mean(1), yte)
    fcast_table["RollLowConf"] = score_baseline_metrics(
        -Xval[:, :, 0].mean(1), yval, -Xte[:, :, 0].mean(1), yte)
    gbm = HistGradientBoostingClassifier(
        max_depth=3, max_iter=150, learning_rate=0.05,
        class_weight="balanced", random_state=0
    ).fit(window_summary_features(Xtr), ytr)
    g_val = gbm.predict_proba(window_summary_features(Xval))[:, 1]
    g_te  = gbm.predict_proba(window_summary_features(Xte))[:, 1]
    fcast_table["GBM-summary"] = score_baseline_metrics(g_val, yval, g_te, yte)

    # ---------------------------------------------------------------- #
    banner("STEP 7 - Train LSTM and GRU")
    mu, sd = fit_norm(Xtr)
    Ntr, Nval, Nte = (Xtr - mu) / sd, (Xval - mu) / sd, (Xte - mu) / sd
    tr_idx = np.arange(0, len(Ntr), cfg["train_stride"])
    Ntr_s, ytr_s = Ntr[tr_idx], ytr[tr_idx]
    v_sub = np.arange(0, len(Nval), 2)
    print(f"  Training on {len(Ntr_s)} windows (stride {cfg['train_stride']}), "
          f"{cfg['epochs']} epochs, {cfg['seeds']} seeds")

    seeds = list(range(cfg["seeds"]))
    cache_dir = os.path.join(ARTIFACTS_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tb_path = os.path.join("results", "table_b_forecasting.json")
    resumed = bool(resume and os.path.exists(tb_path) and all(
        os.path.exists(os.path.join(ARTIFACTS_DIR, f"{c}_params.npz"))
        for c in ("lstm", "gru")))
    best_models, per_seed = {}, {}
    if resumed:
        print("  --resume: reusing trained LSTM/GRU from artifacts/ and "
              "results/table_b_forecasting.json")
        with open(tb_path) as f:
            fcast_table = json.load(f)
        for cell in ("lstm", "gru"):
            m = RNNDriftDetector(cell_type=cell, n_in=Ntr.shape[2],
                                 n_hidden=cfg["n_hidden"], lr=cfg["lr"], seed=0,
                                 pos_weight=pos_weight)
            m.set_params(dict(np.load(os.path.join(ARTIFACTS_DIR, f"{cell}_params.npz"))))
            vprobs, tprobs = predict_all(m, Nval), predict_all(m, Nte)
            thr = float(fcast_table[cell.upper()]["threshold"])
            best_models[cell] = (m, thr, tprobs, vprobs)
            plot_confusion_matrix(np.array(fcast_table[cell.upper()]["confusion_matrix"]),
                                  cell.upper())
            plot_pr_curve(yte, tprobs, cell.upper())
    else:
        for cell in ("lstm", "gru"):
            print(f"\n  [{cell.upper()}]")
            seed_metrics, best_val_ap, best = [], -1.0, None
            for s in seeds:
                t0 = time.perf_counter()
                m, bep, vap = train_rnn(cell, Ntr_s, ytr_s, Nval[v_sub], yval[v_sub],
                                        pos_weight, s, cfg)
                rt = time.perf_counter() - t0
                vprobs, tprobs = predict_all(m, Nval), predict_all(m, Nte)
                thr  = select_threshold(yval, vprobs, metric="f1")
                mets = forecasting_metrics(yte, tprobs, thr)
                mets["runtime_s"], mets["best_epoch"] = rt, int(bep)
                seed_metrics.append(mets)
                print(f"    seed={s} ep*={bep} thr={thr:.2f}  F1={mets['f1_pos']:.3f} "
                      f"P={mets['precision_pos']:.3f} R={mets['recall_pos']:.3f} "
                      f"Acc={mets['accuracy']:.3f} PR-AUC={mets['pr_auc']:.3f} "
                      f"(val AP {vap:.3f})  {rt:.0f}s")
                if vap > best_val_ap:
                    best_val_ap, best = vap, (m, thr, tprobs, vprobs)
            per_seed[cell] = seed_metrics
            keys = ("f1_pos", "recall_pos", "precision_pos", "pr_auc", "accuracy")
            agg = {k: float(np.mean([m[k] for m in seed_metrics])) for k in keys}
            agg.update({f"{k}_std": float(np.std([m[k] for m in seed_metrics]))
                        for k in keys})
            agg["threshold"]  = float(best[1])
            agg["prevalence"] = float(yte.mean())
            agg["confusion_matrix"] = forecasting_metrics(
                yte, best[2], best[1])["confusion_matrix"]
            fcast_table[cell.upper()] = agg
            best_models[cell] = best
            thr_p = select_threshold_precision(yval, best[3], cfg["hp_min_recall"])
            hp = forecasting_metrics(yte, best[2], thr_p)
            hp.pop("confusion_matrix", None)
            fcast_table[cell.upper() + "-hiPrec"] = hp
            plot_confusion_matrix(np.array(agg["confusion_matrix"]), cell.upper())
            plot_pr_curve(yte, best[2], cell.upper())
            np.save(os.path.join(ARTIFACTS_DIR, f"{cell}_best_probs.npy"), best[2])
            np.savez(os.path.join(ARTIFACTS_DIR, f"{cell}_params.npz"),
                     **best[0].get_params())

    print_table_b(fcast_table)
    save_json(fcast_table, "table_b_forecasting.json")

    lstm_probs, gru_probs = best_models["lstm"][2], best_models["gru"][2]
    cte_rel = cte - ws
    plot_degradation_probs(cte_rel, lstm_probs, gru_probs, yte)
    plot_signals_and_detectors(confs, ents, errors, adwin_flags, ddm_flags,
                               lstm_prob=lstm_probs, lstm_centers=cte_rel,
                               gru_prob=gru_probs,   gru_centers=cte_rel)
    plot_model_comparison(fcast_table)

    # ---------------------------------------------------------------- #
    banner("STEP 8 - Monitoring-signal ablation (LSTM)")
    abl = {}
    variants = {"Conf only": [0], "Ent only": [1]}
    for label, cols in variants.items():
        f1s, prs = [], []
        for s in seeds:
            cpath = os.path.join(cache_dir, f"abl_{label.replace(' ', '_')}_s{s}.json")
            if resume and os.path.exists(cpath):
                with open(cpath) as f:
                    c = json.load(f)
                f1s.append(c["f1"]); prs.append(c["pr"])
                continue
            m, _, _ = train_rnn("lstm", Ntr_s[:, :, cols], ytr_s,
                                Nval[v_sub][:, :, cols], yval[v_sub],
                                pos_weight, s, cfg)
            vp = predict_all(m, Nval[:, :, cols])
            tp = predict_all(m, Nte[:, :, cols])
            mt = forecasting_metrics(yte, tp, select_threshold(yval, vp, "f1"))
            f1s.append(mt["f1_pos"]); prs.append(mt["pr_auc"])
            with open(cpath, "w") as f:
                json.dump({"f1": mt["f1_pos"], "pr": mt["pr_auc"]}, f)
        abl[label] = dict(f1_pos=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
                          pr_auc=float(np.mean(prs)), pr_std=float(np.std(prs)))
    abl["Conf+Ent"] = dict(
        f1_pos=fcast_table["LSTM"]["f1_pos"], f1_std=fcast_table["LSTM"]["f1_pos_std"],
        pr_auc=fcast_table["LSTM"]["pr_auc"], pr_std=fcast_table["LSTM"]["pr_auc_std"])
    for k, v in abl.items():
        print(f"  [{k:<9}] F1={v['f1_pos']:.3f} +-{v['f1_std']:.3f}   "
              f"PR-AUC={v['pr_auc']:.3f} +-{v['pr_std']:.3f}")
    save_json(abl, "ablation_results.json")
    plot_ablation(abl)

    # ---------------------------------------------------------------- #
    banner("STEP 9 - Early warning: forecasters vs ADWIN / DDM (test stream)")
    # severe-degradation events (stricter than the training label so that
    # events are sparse enough for a meaningful hit/chance comparison)
    onsets = degradation_onsets(((fe_te - pe_te) > cfg["event_delta"]).astype(int),
                                cte_rel, merge_gap=cfg["merge_gap"])
    cval_rel = cval - ws

    def fore_alarms(cell):
        """Run the adaptive rule over val+test outputs (so the history is warm
        at the start of test) and keep the alarms that fall in the test range."""
        _, _, tprobs, vprobs = best_models[cell]
        allp = np.concatenate([vprobs, tprobs])
        allc = np.concatenate([cval_rel, cte_rel])
        al = alarms_from_probs_adaptive(allp, allc, cfg["alarm_quantile"],
                                        cfg["alarm_history"], 300, cfg["refractory"])
        return al[al >= cte_rel[0]]

    print(f"  Events in test stream: {len(onsets)}   forecaster alarm rule: "
          f"P >= {cfg['alarm_quantile']:.2f}-quantile of own past {cfg['alarm_history']} outputs")
    alarm_sets = {
        "LSTM": fore_alarms("lstm"),
        "GRU":  fore_alarms("gru"),
        "ADWIN":        alarms_from_flags(adwin_flags, 0, cfg["refractory"]),
        "ADWIN(d=.05)": alarms_from_flags(adwin_sens, 0, cfg["refractory"]),
        "DDM":          alarms_from_flags(ddm_flags, 0, cfg["refractory"]),
    }
    ew = compare_methods(onsets, alarm_sets, start=test_start - ws, end=n - ws,
                         lead_max=cfg["lead_max"], tol=cfg["tol"])
    print_table_d(ew)
    save_json(ew, "table_d_early_warning.json")
    plot_early_warning(ew)

    # ---------------------------------------------------------------- #
    banner("STEP 10 - Synthetic drift benchmark (ADWIN vs DDM)")
    det_table = {}
    for seed_s in [42, 7, 99]:
        errs_s, cp = make_synthetic_stream(n_before=2000, n_after=2000,
                                           p_before=0.10, p_after=0.45, seed=seed_s)
        det_s = run_detectors_on_stream(errs_s)
        for dname in ("adwin", "ddm"):
            m = detector_metrics(det_s[dname]["flags"], [cp], total_len=len(errs_s))
            det_table[f"{dname.upper()}_seed{seed_s}"] = {
                **m, "runtime_s": det_s[dname]["runtime_s"],
                "experiment": f"single_cp_seed{seed_s}", "change_points": [int(cp)]}
    errs_rep, cp_rep = make_synthetic_stream(seed=42)
    det_rep = run_detectors_on_stream(errs_rep)
    plot_synthetic_drift_experiment(errs_rep, [cp_rep], det_rep["adwin"]["flags"],
                                    det_rep["ddm"]["flags"])
    for seed_s in [42, 7]:
        errs_nc = make_no_change_stream(n=4000, seed=seed_s)
        det_nc = run_detectors_on_stream(errs_nc)
        for dname in ("adwin", "ddm"):
            m = detector_metrics(det_nc[dname]["flags"], [], total_len=len(errs_nc))
            det_table[f"{dname.upper()}_nochange_seed{seed_s}"] = {
                **m, "runtime_s": det_nc[dname]["runtime_s"],
                "experiment": f"no_change_seed{seed_s}", "change_points": []}
    for dname in ("adwin", "ddm"):
        det_table[f"{dname.upper()}_elec2_real"] = {
            "experiment": "elec2_real_stream", "change_points": "N/A",
            "n_alarms": int(det_out[dname]["flags"].sum()),
            "runtime_s": det_out[dname]["runtime_s"]}
    print_table_c({k: v for k, v in det_table.items()
                   if isinstance(v.get("n_true_detections"), int)})
    save_json(det_table, "table_c_drift_detection.json")

    # ---------------------------------------------------------------- #
    banner("STEP 11 - Save metadata and artifacts")
    meta = dict(cfg)
    meta.update(n_samples=int(n), split_idx={k: int(v) for k, v in si.items()},
                scaler=info["scaler"], pos_weight=float(pos_weight),
                n_train_windows=int(len(Xtr)), n_val_windows=int(len(Xval)),
                n_test_windows=int(len(Xte)),
                pos_rate=dict(train=float(ytr.mean()), val=float(yval.mean()),
                              test=float(yte.mean())),
                seeds=seeds, runtime_s=float(time.perf_counter() - t_start))
    save_json(meta, "experiment_metadata.json")
    for name, arr in dict(confs=confs, ents=ents, errors=errors,
                          centers_test=cte, yte=yte, adwin_flags=adwin_flags,
                          ddm_flags=ddm_flags).items():
        np.save(os.path.join(ARTIFACTS_DIR, f"{name}.npy"), arr)
    print(f"  Done in {time.perf_counter() - t_start:.0f}s")
    print("  Artifacts -> artifacts/   Results -> results/   Figures -> reports/figures/")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1-epoch smoke run")
    ap.add_argument("--resume", action="store_true",
                    help="reuse trained LSTM/GRU + cached ablation runs")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--stride", type=int, help="training-window stride")
    a = ap.parse_args()
    over = dict(CFG)
    if a.epochs: over["epochs"] = a.epochs
    if a.seeds:  over["seeds"] = a.seeds
    if a.stride: over["train_stride"] = a.stride
    main(over, quick=a.quick, resume=a.resume)
