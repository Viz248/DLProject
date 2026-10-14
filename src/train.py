"""
End-to-end pipeline — run from the project root:
    python -m src.train

Execution order
---------------
1.  Load + preprocess Elec2 stream (chronological 60/20/20 split, no leakage)
2.  Warm-up train MLP on training period, then prequential evaluation over
    the full post-warmup stream
3.  Collect MLP monitoring signals: confidence, entropy, error indicator
4.  Run ADWIN and DDM on the error stream (statistical baselines)
5.  Build sliding-window sequence datasets; apply split boundaries
6.  Compute class weight; tune classification threshold on validation set
7.  Train LSTM and GRU degradation forecasters (class-weighted, N_SEEDS seeds)
8.  Evaluate all models on the held-out test set
9.  Monitoring-signal ablation study
10. Controlled synthetic-stream ADWIN/DDM benchmark
11. Save artifacts, plots, and result tables
"""
import os
import json
import time
import numpy as np
from sklearn.metrics import f1_score

from src.preprocessing    import build_stream
from src.mlp              import MLP, run_stream
from src.drift_detectors  import (run_detectors_on_stream,
                                   make_synthetic_stream,
                                   make_no_change_stream,
                                   detector_metrics)
from src.sequence_dataset import make_windows, split_windows
from src.rnn_layers       import RNNDriftDetector
from src.evaluate         import (
    mlp_metrics, forecasting_metrics,
    majority_class_metrics, rolling_mean_baseline_metrics,
    select_threshold,
    save_json, save_numpy,
    plot_mlp_over_time, plot_signals_and_detectors,
    plot_confusion_matrix, plot_pr_curve,
    plot_degradation_probs, plot_synthetic_drift_experiment,
    plot_model_comparison, plot_ablation,
    print_table_a, print_table_b, print_table_c,
)

ARTIFACTS_DIR = "artifacts"
N_SEEDS  = 3
EPOCHS   = 10
N_HIDDEN = 32
LR       = 0.005
WINDOW   = 20
HORIZON  = 20
DELTA    = 0.15


def _ensure():
    for d in (ARTIFACTS_DIR, "results",
              os.path.join("reports", "figures")):
        os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------------ #
def _f1(y, probs, thr):
    return float(f1_score(y, (probs >= thr).astype(int),
                          zero_division=0, pos_label=1))


def train_rnn(cell_type, Xtr, ytr, n_in, pos_weight, seed):
    model = RNNDriftDetector(cell_type=cell_type, n_in=n_in,
                              n_hidden=N_HIDDEN, lr=LR,
                              seed=seed, pos_weight=pos_weight)
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        idx = rng.permutation(len(Xtr))
        for i in idx:
            model.train_step(Xtr[i], int(ytr[i]))
    return model


# ------------------------------------------------------------------ #
def run_ablation(Xtr, ytr, Xval, yval, Xte, yte, pos_weight, seeds):
    """
    Ablation over monitoring-signal subsets.
    Window feature layout: col 0 = confidence, col 1 = entropy.
    """
    n_in_full = Xtr.shape[2]
    variants = {
        "Conf only": [0],
        "Ent only":  [1],
        "Conf+Ent":  [0, 1],
    }
    if n_in_full > 2:
        variants["Full"] = list(range(n_in_full))

    results = {}
    for label, cols in variants.items():
        Xtr_ab  = Xtr[:, :, cols]
        Xval_ab = Xval[:, :, cols]
        Xte_ab  = Xte[:, :, cols]
        seed_f1s, seed_prs = [], []
        for s in seeds:
            m      = train_rnn("lstm", Xtr_ab, ytr, len(cols), pos_weight, s)
            vprobs = np.array([m.predict(w) for w in Xval_ab])
            thr    = select_threshold(yval, vprobs, metric="f1")
            tprobs = np.array([m.predict(w) for w in Xte_ab])
            mets   = forecasting_metrics(yte, tprobs, thr)
            seed_f1s.append(mets["f1_pos"])
            seed_prs.append(mets["pr_auc"])
        results[label] = {
            "f1_pos": float(np.mean(seed_f1s)),
            "f1_std": float(np.std(seed_f1s)),
            "pr_auc": float(np.mean(seed_prs)),
            "pr_std": float(np.std(seed_prs)),
        }
        print(f"  [{label}]  F1={results[label]['f1_pos']:.3f}"
              f" ±{results[label]['f1_std']:.3f}   "
              f"PR-AUC={results[label]['pr_auc']:.3f}"
              f" ±{results[label]['pr_std']:.3f}")
    return results


# ================================================================== #
def main(seed: int = 0):
    _ensure()

    # ---------------------------------------------------------------- #
    print("=" * 62)
    print("STEP 1 — Load and preprocess Elec2 stream")
    print("=" * 62)
    X, y, split_idx, scaler = build_stream(
        "data/electricity.csv",
        train_frac=0.60, val_frac=0.20, temporal_gap=40)
    n          = len(X)
    train_end  = split_idx["train_end"]
    val_start  = split_idx["val_start"]
    val_end    = split_idx["val_end"]
    test_start = split_idx["test_start"]

    print(f"  Total samples : {n}")
    print(f"  Train         : 0–{train_end}  ({train_end} samples)")
    print(f"  Validation    : {val_start}–{val_end}  ({val_end-val_start} samples)")
    print(f"  Test          : {test_start}–{n}  ({n-test_start} samples)")
    print(f"  Label balance : UP={y.mean():.3f}  DOWN={1-y.mean():.3f}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 62)
    print("STEP 2 — MLP warm-up training on training period")
    print("=" * 62)
    mlp = MLP(n_in=X.shape[1], n_hidden=64, n_classes=2,
              lr=0.003, dropout=0.2, seed=seed)
    mlp.fit_batch(X[:train_end], y[:train_end], epochs=30, batch_size=64)
    print("  Warm-up training complete.")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 62)
    print("STEP 3 — Prequential evaluation from train_end")
    print("=" * 62)
    preds, errors, confs, ents = run_stream(
        mlp, X, y, start=train_end, online_update=True)
    stream_len = len(errors)
    print(f"  Post-warmup stream length: {stream_len}")
    print(f"  Overall streaming accuracy: {1 - errors.mean():.4f}")

    # MLP test-set metrics
    # errors/confs/ents are indexed from 0 == stream position train_end
    test_rel   = test_start - train_end   # relative start index for test portion
    y_ts       = y[test_start:]
    preds_ts   = preds[test_rel:]
    confs_ts   = confs[test_rel:]
    # P(class=1): if predicted 1, confidence; else 1 - confidence
    prob1_ts   = np.where(preds_ts == 1, confs_ts, 1.0 - confs_ts)

    mlp_m = mlp_metrics(y_ts, preds_ts, prob1_ts)
    print_table_a(mlp_m)
    save_json(mlp_m, "table_a_mlp_metrics.json")
    plot_mlp_over_time(errors, confs, ents, out_name="mlp_signals.png")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 62)
    print("STEP 4 — ADWIN and DDM on post-warmup error stream")
    print("=" * 62)
    det_out     = run_detectors_on_stream(errors)
    adwin_flags = det_out["adwin"]["flags"]
    ddm_flags   = det_out["ddm"]["flags"]
    print(f"  ADWIN alarms: {adwin_flags.sum()}  "
          f"(runtime {det_out['adwin']['runtime_s']:.3f}s)")
    print(f"  DDM   alarms: {ddm_flags.sum()}  "
          f"(runtime {det_out['ddm']['runtime_s']:.3f}s)")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 62)
    print("STEP 5 — Build sliding-window sequence datasets")
    print("=" * 62)
    Xw, yw, centers, pe, fe = make_windows(
        confs, ents, errors,
        window_len=WINDOW, horizon=HORIZON,
        degradation_delta=DELTA)

    # centers are relative to train_end; convert to absolute
    centers_abs = centers + train_end

    (Xtr, ytr, ctr, pe_tr, _), \
    (Xval, yval, cval, pe_val, _), \
    (Xte, yte, cte, pe_te, _) = split_windows(
        Xw, yw, centers_abs, pe, fe,
        val_start_abs=val_start,
        test_start_abs=test_start)

    print(f"  All windows   : {len(Xw)}  pos={yw.mean():.3f}")
    print(f"  Train windows : {len(Xtr)}  pos={ytr.mean():.3f}")
    print(f"  Val windows   : {len(Xval)}  pos={yval.mean():.3f}")
    print(f"  Test windows  : {len(Xte)}  pos={yte.mean():.3f}")
    n_in = Xw.shape[2]

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 62)
    print("STEP 6 — Class weight + threshold tuning on validation")
    print("=" * 62)
    pos_rate   = float(ytr.mean()) if ytr.mean() > 0 else 0.01
    pos_weight = (1.0 - pos_rate) / pos_rate
    print(f"  Train positive rate: {pos_rate:.3f}  pos_weight={pos_weight:.2f}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 62)
    print("STEP 7 — Train LSTM and GRU (3 seeds each)")
    print("=" * 62)
    seeds        = list(range(N_SEEDS))
    fcast_table  = {}
    best_models  = {}

    for cell_type in ("lstm", "gru"):
        print(f"\n  [{cell_type.upper()}]")
        seed_metrics, all_probs = [], []
        best_val_f1, best_m, best_thr, best_probs = -1.0, None, 0.5, None

        for s in seeds:
            t0    = time.perf_counter()
            m     = train_rnn(cell_type, Xtr, ytr, n_in, pos_weight, s)
            rt    = time.perf_counter() - t0
            vprobs = np.array([m.predict(w) for w in Xval])
            thr    = select_threshold(yval, vprobs, metric="f1")
            tprobs = np.array([m.predict(w) for w in Xte])
            mets   = forecasting_metrics(yte, tprobs, thr)
            mets["runtime_s"] = rt
            seed_metrics.append(mets)
            all_probs.append(tprobs)
            print(f"    seed={s}  thr={thr:.2f}  "
                  f"F1={mets['f1_pos']:.3f}  Rec={mets['recall_pos']:.3f}  "
                  f"Prec={mets['precision_pos']:.3f}  "
                  f"PR-AUC={mets['pr_auc']:.3f}  rt={rt:.1f}s")
            vf1 = _f1(yval, vprobs, thr)
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_m, best_thr, best_probs = m, thr, tprobs

        f1s = [m["f1_pos"]         for m in seed_metrics]
        prs = [m["pr_auc"]         for m in seed_metrics]
        rcs = [m["recall_pos"]     for m in seed_metrics]
        pcs = [m["precision_pos"]  for m in seed_metrics]
        agg = {
            "f1_pos":        float(np.mean(f1s)),
            "f1_std":        float(np.std(f1s)),
            "recall_pos":    float(np.mean(rcs)),
            "rec_std":       float(np.std(rcs)),
            "precision_pos": float(np.mean(pcs)),
            "prec_std":      float(np.std(pcs)),
            "pr_auc":        float(np.mean(prs)),
            "pr_std":        float(np.std(prs)),
            "threshold":     float(best_thr),
            "prevalence":    float(yte.mean()),
            "accuracy":      float(np.mean([m["accuracy"] for m in seed_metrics])),
            "confusion_matrix": forecasting_metrics(
                yte, best_probs, best_thr)["confusion_matrix"],
        }
        fcast_table[cell_type.upper()] = agg
        best_models[cell_type] = (best_m, best_thr, best_probs)

        # per-model plots
        plot_confusion_matrix(np.array(agg["confusion_matrix"]),
                              cell_type.upper())
        plot_pr_curve(yte, best_probs, cell_type.upper())
        np.save(os.path.join(ARTIFACTS_DIR, f"{cell_type}_best_probs.npy"),
                best_probs)
        params_d = best_m.get_params()
        np.savez(os.path.join(ARTIFACTS_DIR, f"{cell_type}_params.npz"),
                 **params_d)

    # Baselines
    maj  = majority_class_metrics(yte)
    fcast_table["Majority"] = maj

    # Rolling-entropy baseline — align entropy to test centers
    ent_test_aligned = ents[cte - train_end]
    roll_base = rolling_mean_baseline_metrics(ent_test_aligned, yte)
    fcast_table["RollingEnt"] = roll_base

    print_table_b(fcast_table)
    save_json(fcast_table, "table_b_forecasting.json")

    # Combined plots
    lstm_probs = best_models["lstm"][2]
    gru_probs  = best_models["gru"][2]
    cte_rel    = cte - train_end   # relative stream index for plots
    plot_degradation_probs(cte_rel, lstm_probs, gru_probs, yte)
    plot_signals_and_detectors(confs, ents, errors, adwin_flags, ddm_flags,
                                lstm_prob=lstm_probs, lstm_centers=cte_rel,
                                gru_prob=gru_probs,   gru_centers=cte_rel)
    plot_model_comparison(fcast_table)

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 62)
    print("STEP 8 — Monitoring-signal ablation study")
    print("=" * 62)
    abl = run_ablation(Xtr, ytr, Xval, yval, Xte, yte, pos_weight, seeds)
    save_json(abl, "ablation_results.json")
    plot_ablation(abl)

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 62)
    print("STEP 9 — Synthetic drift benchmark (ADWIN vs DDM)")
    print("=" * 62)
    det_table = {}

    # Single change-point experiment across 3 seeds
    for seed_s in [42, 7, 99]:
        errs_s, cp = make_synthetic_stream(
            n_before=2000, n_after=2000,
            p_before=0.10, p_after=0.45, seed=seed_s)
        det_s = run_detectors_on_stream(errs_s)
        for dname in ("adwin", "ddm"):
            m   = detector_metrics(det_s[dname]["flags"], [cp],
                                   total_len=len(errs_s))
            key = f"{dname.upper()}_seed{seed_s}"
            det_table[key] = {**m,
                "runtime_s":  det_s[dname]["runtime_s"],
                "experiment": f"single_cp_seed{seed_s}",
                "change_points": [int(cp)],
            }

    # Plot representative synthetic run
    errs_rep, cp_rep = make_synthetic_stream(seed=42)
    det_rep = run_detectors_on_stream(errs_rep)
    plot_synthetic_drift_experiment(
        errs_rep, [cp_rep],
        det_rep["adwin"]["flags"], det_rep["ddm"]["flags"])

    # No-change control
    for seed_s in [42, 7]:
        errs_nc = make_no_change_stream(n=4000, seed=seed_s)
        det_nc  = run_detectors_on_stream(errs_nc)
        for dname in ("adwin", "ddm"):
            m   = detector_metrics(det_nc[dname]["flags"], [],
                                   total_len=len(errs_nc))
            key = f"{dname.upper()}_nochange_seed{seed_s}"
            det_table[key] = {**m,
                "runtime_s":  det_nc[dname]["runtime_s"],
                "experiment": f"no_change_seed{seed_s}",
                "change_points": [],
            }

    # Real Elec2 stream — no fabricated ground truth
    for dname in ("adwin", "ddm"):
        flags_real = det_out[dname]["flags"]
        det_table[f"{dname.upper()}_elec2_real"] = {
            "experiment":        "elec2_real_stream",
            "change_points":     "N/A",
            "n_alarms":          int(flags_real.sum()),
            "n_true_detections": "N/A",
            "n_missed":          "N/A",
            "n_false_alarms":    "N/A",
            "mean_delay":        "N/A",
            "detection_rate":    "N/A",
            "false_alarm_rate":  "N/A",
            "runtime_s":         det_out[dname]["runtime_s"],
        }

    # Print only rows with numeric detection metrics
    printable = {k: v for k, v in det_table.items()
                 if isinstance(v.get("n_true_detections"), int)}
    print_table_c(printable)
    save_json(det_table, "table_c_drift_detection.json")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 62)
    print("STEP 10 — Save experiment metadata and final artifacts")
    print("=" * 62)
    meta = {
        "dataset":           "electricity.csv",
        "n_samples":         int(n),
        "split_idx":         {k: int(v) for k, v in split_idx.items()},
        "window_len":        WINDOW,
        "horizon":           HORIZON,
        "degradation_delta": DELTA,
        "n_hidden":          N_HIDDEN,
        "lr":                LR,
        "epochs":            EPOCHS,
        "seeds":             seeds,
        "pos_weight":        float(pos_weight),
    }
    save_json(meta, "experiment_metadata.json")
    np.save(os.path.join(ARTIFACTS_DIR, "confs.npy"),        confs)
    np.save(os.path.join(ARTIFACTS_DIR, "ents.npy"),         ents)
    np.save(os.path.join(ARTIFACTS_DIR, "errors.npy"),       errors)
    np.save(os.path.join(ARTIFACTS_DIR, "centers_test.npy"), cte)
    np.save(os.path.join(ARTIFACTS_DIR, "yte.npy"),          yte)

    print("  Artifacts → artifacts/")
    print("  Results   → results/")
    print("  Figures   → reports/figures/")
    print("\n✓  Pipeline complete.")
    return meta


if __name__ == "__main__":
    main()
