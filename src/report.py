"""
Print the final results of every algorithm in a few plain tables.

    python -m src.report

Reads the committed result snapshots (docs/results/ and docs/novelty/), so it
prints the final numbers without retraining anything.
"""
import json
import os

MAIN = os.path.join("docs", "results")
NOVELTY = os.path.join("docs", "novelty", "novelty_riskgate.json")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _f1(p, r):
    return 0.0 if p != p or not (p + r) else 2 * p * r / (p + r)


def _rule(width=78):
    print("-" * width)


def main():
    meta = _load(os.path.join(MAIN, "experiment_metadata.json"))
    mlp = _load(os.path.join(MAIN, "table_a_mlp_metrics.json"))
    fc = _load(os.path.join(MAIN, "table_b_forecasting.json"))
    ew = _load(os.path.join(MAIN, "table_d_early_warning.json"))
    syn = _load(os.path.join(MAIN, "table_c_drift_detection.json"))
    nov = _load(NOVELTY)["methods"]

    print("=" * 78)
    print("EARLY DRIFT DETECTION - FINAL RESULTS (Elec2 electricity data, test part)")
    print("=" * 78)

    # 1. the model being monitored
    print("\n1. THE MODEL WE MONITOR (MLP predicting price up/down)")
    _rule()
    print(f"   Accuracy {mlp['accuracy']:.1%}   F1 {mlp['f1']:.3f}   "
          f"(on {meta['n_test_windows']:,} test records)")

    # 2. forecasting
    print("\n2. PREDICTING TROUBLE: will the error rate jump in the next 20 steps?")
    print(f"   {fc['LSTM']['prevalence']:.1%} of test moments are followed by a jump (chance level).")
    _rule()
    print(f"   {'Method':<22}{'Type':<10}{'Precision':>10}{'Recall':>8}{'F1':>7}{'PR-AUC':>8}{'Accuracy':>10}")
    rows = [("Always warn", "baseline", "AlwaysWarn"),
            ("Gradient boosting", "baseline", "GBM-summary"),
            ("LSTM", "ours", "LSTM"),
            ("GRU", "ours", "GRU")]
    for name, kind, key in rows:
        m = fc[key]
        print(f"   {name:<22}{kind:<10}{m['precision_pos']:>10.3f}{m['recall_pos']:>8.3f}"
              f"{m['f1_pos']:>7.3f}{m['pr_auc']:>8.3f}{m['accuracy']:>10.3f}")
    print("   -> LSTM/GRU are about 2x chance precision, but no better than gradient boosting.")

    # 3. event detection
    n_ev = int(nov["RiskGate-CUSUM"]["n_events"])
    print(f"\n3. CATCHING REAL DEGRADATION EVENTS ({n_ev} sharp error jumps in the test data)")
    print("   Caught = the event had an alarm from 30 steps before to 60 steps after it began.")
    print("   Delay  = median steps from the start of the event to the alarm (negative = early).")
    _rule()
    print(f"   {'Method':<18}{'Type':<10}{'Alarms':>7}{'Caught':>8}{'Precision':>10}{'F1':>7}"
          f"{'Delay':>7}{'False/1000':>11}")

    def row(name, kind, m):
        p, r = m["alarm_precision"], m["event_recall"]
        f1 = m.get("event_f1", _f1(p, r))
        delay = m.get("median_offset")
        caught = f"{round(r * n_ev)}/{n_ev}"
        prec = "-" if p != p else f"{p:.2f}"
        dl = "-" if delay is None else f"{delay:+.0f}"
        print(f"   {name:<18}{kind:<10}{int(m['n_alarms']):>7}{caught:>8}{prec:>10}{f1:>7.3f}"
              f"{dl:>7}{m['false_alarms_per_1000']:>11.1f}")

    row("ADWIN", "existing", nov["ADWIN"])
    row("DDM", "existing", nov["DDM"])
    row("CUSUM", "existing", nov["CUSUM (plain)"])
    row("RiskGate-CUSUM", "ours", nov["RiskGate-CUSUM"])
    row("LSTM forecaster", "ours", ew["LSTM"])
    row("GRU forecaster", "ours", ew["GRU"])
    print("   -> RiskGate has the best F1 (fewest false alarms), but only ~1 alarm better")
    print("      than plain CUSUM, so the gain is small. Only the forecasters regularly fire early,")
    print("      and they are barely better than random alarms. ADWIN/DDM almost never fire.")

    # 4. synthetic benchmark
    def summary(det):
        d = [v["mean_delay"] for k, v in syn.items()
             if k.startswith(det + "_seed") and v.get("mean_delay") is not None]
        fa = sum(v["n_false_alarms"] for k, v in syn.items()
                 if k.startswith(det + "_") and isinstance(v.get("n_false_alarms"), int))
        return min(d), max(d), fa

    print("\n4. CONTROLLED TEST: error rate jumps from 10% to 45% at a known point")
    _rule()
    for det in ("ADWIN", "DDM"):
        lo, hi, fa = summary(det)
        print(f"   {det:<6} caught every change, {lo:.0f}-{hi:.0f} steps late, {fa} false alarm(s)")
    print("   -> both work on big, lasting changes; ADWIN is faster.")
    print()


if __name__ == "__main__":
    main()
